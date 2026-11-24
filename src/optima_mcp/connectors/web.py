"""Web connectors: Google Sheets (auto CSV export) and REST APIs (GET/POST + JSON flattening)."""

from __future__ import annotations

import io
import re
from urllib.parse import parse_qs, urlparse

import httpx
import pandas as pd

from optima_mcp.connectors.base import BaseConnector, DataSourceConfig, resolve_env_value
from optima_mcp.connectors.files import json_payload_to_dataframe

_GSHEET_ID_RE = re.compile(r"/spreadsheets/d/([a-zA-Z0-9-_]+)")
_GID_RE = re.compile(r"gid=(\d+)")


def transform_google_sheets_url(url: str) -> str:
    """Transform a Google Sheets `/edit...` (or share) URL into a direct CSV export URL.

    Preserves `gid` from either query parameters (`?gid=123`) or fragment (`#gid=123`).
    """
    parsed = urlparse(url.strip())
    if "docs.google.com" not in parsed.netloc.lower() or "/spreadsheets/d/" not in parsed.path:
        raise ValueError(
            f"Invalid Google Sheets URL: '{url}'. Expected 'https://docs.google.com/spreadsheets/d/<ID>/...'."
        )

    match = _GSHEET_ID_RE.search(parsed.path)
    if not match:
        raise ValueError(f"Could not extract spreadsheet ID from URL: '{url}'.")
    sheet_id = match.group(1)

    # Check if already export?format=csv
    query_params = parse_qs(parsed.query)
    gid: str | None = None
    if "gid" in query_params and query_params["gid"]:
        gid = query_params["gid"][0]
    elif parsed.fragment:
        frag_match = _GID_RE.search(parsed.fragment)
        if frag_match:
            gid = frag_match.group(1)

    export_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    if gid is not None:
        export_url = f"{export_url}&gid={gid}"
    return export_url


class GoogleSheetsConnector(BaseConnector):
    """Fetches public Google Sheets directly as CSV into Pandas."""

    def load(self, config: DataSourceConfig) -> pd.DataFrame:
        if not config.url:
            raise ValueError("GoogleSheetsConnector requires 'url' in source_config.")

        export_url = transform_google_sheets_url(config.url)
        headers = {k: resolve_env_value(v) for k, v in (config.headers or {}).items()}

        with httpx.Client(follow_redirects=True, timeout=config.timeout_seconds) as client:
            response = client.get(export_url, headers=headers)
            response.raise_for_status()

        return pd.read_csv(io.StringIO(response.text))


class RestApiConnector(BaseConnector):
    """Fetches JSON from REST APIs (GET/POST) with env-resolved auth headers & dot-path flattening."""

    def load(self, config: DataSourceConfig) -> pd.DataFrame:
        if not config.url:
            raise ValueError("RestApiConnector requires 'url' in source_config.")

        headers = {k: resolve_env_value(v) for k, v in (config.headers or {}).items()}

        with httpx.Client(follow_redirects=True, timeout=config.timeout_seconds) as client:
            if config.method == "POST":
                response = client.post(
                    config.url,
                    headers=headers,
                    params=config.params,
                    json=config.body,
                )
            else:
                response = client.get(
                    config.url,
                    headers=headers,
                    params=config.params,
                )
            response.raise_for_status()
            payload = response.json()

        return json_payload_to_dataframe(payload, data_path=config.data_path)
