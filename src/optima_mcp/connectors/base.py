"""Unified DataSourceConfig, ConnectorRegistry, and zero-context-flooding inspector."""

from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

SourceType = Literal[
    "auto",
    "csv",
    "excel",
    "json",
    "parquet",
    "sql",
    "google_sheets",
    "rest_api",
]

_AUTH_ENV_RE = re.compile(
    r"^(?P<prefix>(?:Bearer|Token|Basic|ApiKey)\s+)env:(?P<var>[A-Za-z_][A-Za-z0-9_]*)$",
    re.IGNORECASE,
)


def resolve_env_value(value: str) -> str:
    """Resolve `env:VAR_NAME` (or `<AuthScheme> env:VAR_NAME`) from local environment variables.

    Secrets stay in the local environment and never appear in LLM transcripts.
    """
    stripped = value.strip()
    if stripped.startswith("env:"):
        var_name = stripped[4:].strip()
        if not var_name:
            raise ValueError("Empty environment variable name after 'env:' prefix.")
        resolved = os.environ.get(var_name)
        if resolved is None:
            raise ValueError(f"Environment variable '{var_name}' is not set.")
        return resolved

    match = _AUTH_ENV_RE.match(stripped)
    if match:
        var_name = match.group("var")
        resolved = os.environ.get(var_name)
        if resolved is None:
            raise ValueError(f"Environment variable '{var_name}' is not set.")
        return f"{match.group('prefix')}{resolved}"

    return value


class DataSourceConfig(BaseModel):
    """Unified configuration for connecting to any supported data source."""

    source_type: SourceType = Field(
        default="auto",
        description=(
            "Connector type: 'csv', 'excel', 'json', 'parquet', 'sql', "
            "'google_sheets', 'rest_api', or 'auto' to infer from path/url/connection_string."
        ),
    )
    # Local file options
    path: str | None = Field(default=None, description="Path to local CSV, Excel, JSON, or Parquet file.")
    sheet_name: str | int | None = Field(default=0, description="Excel sheet name or 0-based index.")
    encoding: str | None = Field(default=None, description="Optional text encoding for CSV files.")
    delimiter: str | None = Field(default=None, description="Optional delimiter for CSV files.")

    # SQL options
    connection_string: str | None = Field(
        default=None,
        description="SQLAlchemy connection string or 'env:VAR_NAME' reference.",
    )
    query: str | None = Field(
        default=None,
        description="Read-only SELECT/WITH SQL query.",
    )

    # Web / REST / Google Sheets options
    url: str | None = Field(default=None, description="Google Sheets URL or REST API endpoint URL.")
    method: Literal["GET", "POST"] = Field(default="GET", description="HTTP method for REST APIs.")
    headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers (values may use 'env:VAR_NAME').",
    )
    params: dict[str, Any] | None = Field(default=None, description="Optional query parameters for REST APIs.")
    body: dict[str, Any] | None = Field(default=None, description="Optional JSON body for POST REST APIs.")
    data_path: str | None = Field(
        default=None,
        description="Optional dot-notation path to extract nested JSON arrays (e.g., 'data.items').",
    )
    timeout_seconds: float = Field(default=30.0, gt=0, le=300.0, description="HTTP request timeout in seconds.")

    def resolved_source_type(self) -> str:
        """Infer concrete source type when `source_type == 'auto'`."""
        if self.source_type != "auto":
            return self.source_type

        if self.connection_string or self.query:
            return "sql"

        if self.url:
            if "docs.google.com/spreadsheets" in self.url.lower():
                return "google_sheets"
            return "rest_api"

        if self.path:
            ext = Path(self.path).suffix.lower()
            ext_map = {
                ".csv": "csv",
                ".tsv": "csv",
                ".xlsx": "excel",
                ".xls": "excel",
                ".json": "json",
                ".parquet": "parquet",
                ".pq": "parquet",
            }
            if ext in ext_map:
                return ext_map[ext]
            raise ValueError(
                f"Cannot infer source_type from file extension '{ext}'. "
                "Specify source_type explicitly ('csv', 'excel', 'json', 'parquet')."
            )

        raise ValueError(
            "Unable to infer source_type: provide 'path', 'url', or 'connection_string'+'query'."
        )


class BaseConnector(ABC):
    """Abstract base class for all Optima-MCP data connectors."""

    @abstractmethod
    def load(self, config: DataSourceConfig) -> pd.DataFrame:
        """Load the full dataset locally into a Pandas DataFrame."""


def _to_json_safe(value: Any, max_str_len: int = 200) -> Any:
    """Convert numpy/pandas values into compact, JSON-serializable primitives."""
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        fval = float(value)
        if math.isnan(fval) or math.isinf(fval):
            return None
        return round(fval, 6)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (list, dict)):
        text = str(value)
        return text if len(text) <= max_str_len else text[:max_str_len] + "..."
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    text = str(value)
    return text if len(text) <= max_str_len else text[:max_str_len] + "..."


def inspect_dataframe(df: pd.DataFrame, max_sample_rows: int = 3) -> dict[str, Any]:
    """Return schema, null counts, numeric summary, and at most 3 sample rows.

    Enforces zero-context-flooding: raw tables are NEVER returned to the LLM.
    """
    sample_limit = max(0, min(max_sample_rows, 3))
    columns = [str(c) for c in df.columns]
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    null_counts = {str(c): int(df[c].isna().sum()) for c in df.columns}

    sample_rows: list[dict[str, Any]] = []
    if len(df) > 0 and sample_limit > 0:
        for _, row in df.head(sample_limit).iterrows():
            sample_rows.append({str(col): _to_json_safe(row[col]) for col in df.columns})

    numeric_summary: dict[str, dict[str, float | None]] = {}
    for col in df.columns:
        series = df[col]
        if pd.api.types.is_bool_dtype(series):
            continue
        if pd.api.types.is_numeric_dtype(series):
            clean = series.dropna()
            if len(clean) > 0:
                numeric_summary[str(col)] = {
                    "min": _to_json_safe(clean.min()),
                    "max": _to_json_safe(clean.max()),
                    "mean": _to_json_safe(clean.mean()),
                }
            else:
                numeric_summary[str(col)] = {"min": None, "max": None, "mean": None}

    return {
        "row_count": int(len(df)),
        "columns": columns,
        "dtypes": dtypes,
        "null_counts": null_counts,
        "sample_rows": sample_rows,
        "numeric_summary": numeric_summary,
    }


class ConnectorRegistry:
    """Central registry mapping DataSourceConfig types to concrete connectors."""

    def __init__(self) -> None:
        self._connectors: dict[str, BaseConnector] = {}

    def register(self, source_type: str, connector: BaseConnector) -> None:
        self._connectors[source_type] = connector

    def get(self, source_type: str) -> BaseConnector:
        if source_type not in self._connectors:
            supported = ", ".join(sorted(self._connectors.keys()))
            raise ValueError(
                f"Unsupported source_type '{source_type}'. Supported types: {supported}"
            )
        return self._connectors[source_type]

    def load(self, source_config: dict[str, Any] | DataSourceConfig) -> pd.DataFrame:
        config = (
            source_config
            if isinstance(source_config, DataSourceConfig)
            else DataSourceConfig.model_validate(source_config)
        )
        resolved = config.resolved_source_type()
        connector = self.get(resolved)
        df = connector.load(config)
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"Connector '{resolved}' did not return a pandas DataFrame.")
        return df


def _build_default_registry() -> ConnectorRegistry:
    from optima_mcp.connectors.files import FileConnector
    from optima_mcp.connectors.sql import SQLConnector
    from optima_mcp.connectors.web import GoogleSheetsConnector, RestApiConnector

    registry = ConnectorRegistry()
    file_conn = FileConnector()
    for file_type in ("csv", "excel", "json", "parquet"):
        registry.register(file_type, file_conn)
    registry.register("sql", SQLConnector())
    registry.register("google_sheets", GoogleSheetsConnector())
    registry.register("rest_api", RestApiConnector())
    return registry


_DEFAULT_REGISTRY: ConnectorRegistry | None = None


def get_default_registry() -> ConnectorRegistry:
    """Return the singleton default ConnectorRegistry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _build_default_registry()
    return _DEFAULT_REGISTRY


def load_dataframe(source_config: dict[str, Any] | DataSourceConfig) -> pd.DataFrame:
    """Convenience helper to load a DataFrame using the default ConnectorRegistry."""
    return get_default_registry().load(source_config)
