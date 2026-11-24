"""Universal zero-context-flooding data connectors for Optima-MCP."""

from optima_mcp.connectors.base import (
    BaseConnector,
    ConnectorRegistry,
    DataSourceConfig,
    inspect_dataframe,
    load_dataframe,
    resolve_env_value,
)
from optima_mcp.connectors.files import FileConnector
from optima_mcp.connectors.sql import SQLConnector, SQLSecurityError, validate_readonly_sql
from optima_mcp.connectors.web import (
    GoogleSheetsConnector,
    RestApiConnector,
    transform_google_sheets_url,
)

__all__ = [
    "BaseConnector",
    "ConnectorRegistry",
    "DataSourceConfig",
    "FileConnector",
    "GoogleSheetsConnector",
    "RestApiConnector",
    "SQLConnector",
    "SQLSecurityError",
    "inspect_dataframe",
    "load_dataframe",
    "resolve_env_value",
    "transform_google_sheets_url",
    "validate_readonly_sql",
]
