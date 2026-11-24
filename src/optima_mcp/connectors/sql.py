"""Strict read-only SQL connector (SQLite, PostgreSQL, MySQL) via SQLAlchemy."""

from __future__ import annotations

import re

import pandas as pd
from sqlalchemy import create_engine, text

from optima_mcp.connectors.base import BaseConnector, DataSourceConfig, resolve_env_value

FORBIDDEN_SQL_TOKENS: frozenset[str] = frozenset(
    {
        "INSERT",
        "UPDATE",
        "DELETE",
        "DROP",
        "ALTER",
        "TRUNCATE",
        "CREATE",
        "ATTACH",
        "DETACH",
        "PRAGMA",
        "EXEC",
        "EXECUTE",
        "MERGE",
        "REPLACE",
        "GRANT",
        "REVOKE",
        "VACUUM",
        "REINDEX",
        "CALL",
        "COPY",
        "LOAD",
    }
)

_ALLOWED_START_TOKENS: frozenset[str] = frozenset({"SELECT", "WITH"})
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class SQLSecurityError(ValueError):
    """Raised when a SQL query violates the strict read-only SELECT guard."""


def _strip_comments_and_extract_code(query: str) -> tuple[str, list[str]]:
    """Strip SQL comments and return (cleaned_sql, outside_literal_word_tokens).

    Preserves string literals and quoted identifiers while ignoring their contents
    for forbidden-keyword scanning, and rejects unclosed comments or multi-statements.
    """
    n = len(query)
    i = 0
    cleaned_chars: list[str] = []
    non_literal_chars: list[str] = []
    semicolon_positions: list[int] = []

    while i < n:
        ch = query[i]
        nxt = query[i + 1] if i + 1 < n else ""

        # Single-line comment: -- ...
        if ch == "-" and nxt == "-":
            i += 2
            while i < n and query[i] != "\n":
                i += 1
            cleaned_chars.append(" ")
            non_literal_chars.append(" ")
            continue

        # Block comment: /* ... */
        if ch == "/" and nxt == "*":
            i += 2
            closed = False
            while i + 1 < n:
                if query[i] == "*" and query[i + 1] == "/":
                    i += 2
                    closed = True
                    break
                i += 1
            if not closed:
                raise SQLSecurityError("Unclosed SQL block comment '/*' detected.")
            cleaned_chars.append(" ")
            non_literal_chars.append(" ")
            continue

        # Single-quoted string literal '...'
        if ch == "'":
            cleaned_chars.append(ch)
            non_literal_chars.append(" ")
            i += 1
            closed = False
            while i < n:
                cleaned_chars.append(query[i])
                if query[i] == "'":
                    if i + 1 < n and query[i + 1] == "'":
                        cleaned_chars.append(query[i + 1])
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                i += 1
            if not closed:
                raise SQLSecurityError("Unclosed single-quoted SQL string literal.")
            continue

        # Double-quoted identifier/string "..."
        if ch == '"':
            cleaned_chars.append(ch)
            non_literal_chars.append(" ")
            i += 1
            closed = False
            while i < n:
                cleaned_chars.append(query[i])
                if query[i] == '"':
                    if i + 1 < n and query[i + 1] == '"':
                        cleaned_chars.append(query[i + 1])
                        i += 2
                        continue
                    i += 1
                    closed = True
                    break
                i += 1
            if not closed:
                raise SQLSecurityError("Unclosed double-quoted SQL identifier.")
            continue

        # Backtick-quoted identifier `...`
        if ch == "`":
            cleaned_chars.append(ch)
            non_literal_chars.append(" ")
            i += 1
            closed = False
            while i < n:
                cleaned_chars.append(query[i])
                if query[i] == "`":
                    i += 1
                    closed = True
                    break
                i += 1
            if not closed:
                raise SQLSecurityError("Unclosed backtick-quoted SQL identifier.")
            continue

        if ch == ";":
            semicolon_positions.append(len(cleaned_chars))

        cleaned_chars.append(ch)
        non_literal_chars.append(ch)
        i += 1

    cleaned_sql = "".join(cleaned_chars).strip()
    non_literal_sql = "".join(non_literal_chars).strip()

    # Check semicolons strictly: forbid multi-statement or multiple semicolons
    if semicolon_positions:
        if len(semicolon_positions) > 1:
            raise SQLSecurityError("Multi-statement SQL queries (multiple ';') are forbidden.")
        # If there is any non-whitespace character after the single semicolon, reject
        after_semi = "".join(cleaned_chars[semicolon_positions[0] + 1 :]).strip()
        if after_semi:
            raise SQLSecurityError(
                "Multi-statement SQL queries (tokens after ';') are strictly forbidden."
            )
        # Strip the single harmless trailing semicolon
        cleaned_sql = cleaned_sql.rstrip(";").strip()
        non_literal_sql = non_literal_sql.rstrip(";").strip()

    tokens = [tok.upper() for tok in _WORD_RE.findall(non_literal_sql)]
    return cleaned_sql, tokens


def validate_readonly_sql(query: str) -> str:
    """Validate that a SQL query is strictly read-only SELECT or WITH.

    Strips comments, blocks multi-statement queries, and rejects all mutating/DDL tokens.
    Returns the sanitized SQL query string.
    """
    if not query or not query.strip():
        raise SQLSecurityError("SQL query must not be empty.")

    cleaned_sql, tokens = _strip_comments_and_extract_code(query)
    if not tokens:
        raise SQLSecurityError("SQL query contains no executable tokens after stripping comments.")

    first_token = tokens[0]
    if first_token not in _ALLOWED_START_TOKENS:
        raise SQLSecurityError(
            f"Read-only SQL guard rejected query starting with '{first_token}'. "
            "Only SELECT and WITH queries are permitted."
        )

    for tok in tokens:
        if tok in FORBIDDEN_SQL_TOKENS:
            raise SQLSecurityError(
                f"Read-only SQL guard blocked forbidden mutating/DDL token '{tok}'."
            )

    # Guard against SELECT ... INTO <new_table>
    if "INTO" in tokens:
        raise SQLSecurityError(
            "Read-only SQL guard blocked 'INTO' clause (SELECT INTO table creation is forbidden)."
        )

    return cleaned_sql


class SQLConnector(BaseConnector):
    """Read-only SQLAlchemy connector for SQLite, PostgreSQL, and MySQL."""

    def load(self, config: DataSourceConfig) -> pd.DataFrame:
        if not config.connection_string:
            raise ValueError("SQLConnector requires 'connection_string' in source_config.")
        if not config.query:
            raise ValueError("SQLConnector requires 'query' in source_config.")

        safe_query = validate_readonly_sql(config.query)
        conn_url = resolve_env_value(config.connection_string)

        engine = create_engine(conn_url)
        try:
            with engine.connect() as conn:
                df = pd.read_sql_query(text(safe_query), conn)
                conn.rollback()
            return df
        finally:
            engine.dispose()
