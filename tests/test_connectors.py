"""Tests for universal data connectors, zero-context-flooding inspection, and SQL security."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from optima_mcp.connectors.base import inspect_dataframe, load_dataframe, resolve_env_value
from optima_mcp.connectors.sql import SQLSecurityError, validate_readonly_sql
from optima_mcp.connectors.web import transform_google_sheets_url

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def test_csv_connector_and_zero_context_flooding(tmp_path: Path) -> None:
    # Create a 50-row CSV to verify inspect_dataframe never returns more than 3 rows
    df_in = pd.DataFrame(
        {
            "sku": [f"SKU-{i:03d}" for i in range(50)],
            "cost": [10.0 + i for i in range(50)],
            "stock": [None if i % 10 == 0 else i * 2 for i in range(50)],
        }
    )
    csv_path = tmp_path / "inventory.csv"
    df_in.to_csv(csv_path, index=False)

    df_loaded = load_dataframe({"path": str(csv_path)})
    assert len(df_loaded) == 50

    summary = inspect_dataframe(df_loaded, max_sample_rows=10)  # Even if caller asks for 10, capped at 3
    assert summary["row_count"] == 50
    assert summary["columns"] == ["sku", "cost", "stock"]
    assert len(summary["sample_rows"]) == 3
    assert summary["null_counts"]["stock"] == 5
    assert summary["numeric_summary"]["cost"]["min"] == 10.0
    assert summary["numeric_summary"]["cost"]["max"] == 59.0


def test_excel_and_parquet_connectors(tmp_path: Path) -> None:
    df_in = pd.DataFrame({"item": ["A", "B", "C"], "price": [12.5, 25.0, 40.0]})

    xlsx_path = tmp_path / "prices.xlsx"
    df_in.to_excel(xlsx_path, index=False, sheet_name="Pricing")
    df_xlsx = load_dataframe({"path": str(xlsx_path), "sheet_name": "Pricing"})
    assert list(df_xlsx["item"]) == ["A", "B", "C"]
    assert list(df_xlsx["price"]) == [12.5, 25.0, 40.0]

    pq_path = tmp_path / "prices.parquet"
    df_in.to_parquet(pq_path, index=False)
    df_pq = load_dataframe({"path": str(pq_path)})
    assert len(df_pq) == 3
    assert df_pq["price"].sum() == pytest.approx(77.5)


def test_json_connector_nested_and_example_vrp(tmp_path: Path) -> None:
    nested = {
        "meta": {"version": 1},
        "payload": {
            "items": [
                {"id": "X1", "metrics": {"cost": 5.0, "price": 9.0}},
                {"id": "X2", "metrics": {"cost": 8.0, "price": 15.0}},
            ]
        },
    }
    json_path = tmp_path / "nested.json"
    json_path.write_text(json.dumps(nested), encoding="utf-8")

    df = load_dataframe({"path": str(json_path), "data_path": "payload.items"})
    assert len(df) == 2
    assert "metrics.cost" in df.columns
    assert list(df["id"]) == ["X1", "X2"]

    # Auto-loading structured VRP example JSON extracts `stops`
    vrp_df = load_dataframe({"path": str(EXAMPLES_DIR / "fleet_delivery_orders.json")})
    assert len(vrp_df) == 12
    assert {"stop_id", "lat", "lon", "demand_kg"}.issubset(set(vrp_df.columns))


def test_sqlite_readonly_select_and_env_connection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "store.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE products (sku TEXT, cost REAL, price REAL, note TEXT)")
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?)",
        [
            ("P1", 10.0, 20.0, "Normal item"),
            ("P2", 15.0, 30.0, "No DROP here in string literal"),
        ],
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("OPTIMA_TEST_DB", f"sqlite:///{db_path}")

    # Standard SELECT with string literal containing forbidden word should be ALLOWED
    df = load_dataframe(
        {
            "source_type": "sql",
            "connection_string": "env:OPTIMA_TEST_DB",
            "query": "SELECT sku, cost, price FROM products WHERE note != 'DROP';",
        }
    )
    assert len(df) == 2

    # CTE WITH query should be ALLOWED
    df_cte = load_dataframe(
        {
            "source_type": "sql",
            "connection_string": "env:OPTIMA_TEST_DB",
            "query": "/* fetch margins */ WITH m AS (SELECT sku, price - cost AS margin FROM products) SELECT * FROM m",
        }
    )
    assert list(df_cte["margin"]) == [10.0, 15.0]


@pytest.mark.parametrize(
    "malicious_sql",
    [
        "INSERT INTO products VALUES ('X', 1, 2, '')",
        "UPDATE products SET cost = 0",
        "DELETE FROM products",
        "DROP TABLE products",
        "ALTER TABLE products ADD COLUMN x INT",
        "TRUNCATE TABLE products",
        "CREATE TABLE evil (id INT)",
        "ATTACH DATABASE '/tmp/x.db' AS evil",
        "PRAGMA table_info(products)",
        "EXEC sp_executesql N'SELECT 1'",
        "SELECT * FROM products; DROP TABLE products",
        "SELECT 1; SELECT 2",
        "/* leading comment */ DELETE FROM products",
        "WITH x AS (SELECT 1) DELETE FROM products",
        "SELECT * FROM products INTO backup_table",
        "SELECT * FROM products /* unclosed comment",
        "   ",
    ],
)
def test_sql_write_blocking_security_guard(malicious_sql: str) -> None:
    with pytest.raises(SQLSecurityError):
        validate_readonly_sql(malicious_sql)


def test_google_sheets_url_transform_and_mocked_fetch() -> None:
    assert (
        transform_google_sheets_url(
            "https://docs.google.com/spreadsheets/d/1AbC_xYz-99/edit?usp=sharing#gid=456789"
        )
        == "https://docs.google.com/spreadsheets/d/1AbC_xYz-99/export?format=csv&gid=456789"
    )
    assert (
        transform_google_sheets_url("https://docs.google.com/spreadsheets/d/SHEET123/edit?gid=12")
        == "https://docs.google.com/spreadsheets/d/SHEET123/export?format=csv&gid=12"
    )
    assert (
        transform_google_sheets_url("https://docs.google.com/spreadsheets/d/SHEET123/edit")
        == "https://docs.google.com/spreadsheets/d/SHEET123/export?format=csv"
    )

    with pytest.raises(ValueError):
        transform_google_sheets_url("https://example.com/not-a-sheet")

    mock_resp = MagicMock()
    mock_resp.text = "sku,cost,price\nA1,10,20\nB2,15,28\n"
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.get.return_value = mock_resp

    with patch("optima_mcp.connectors.web.httpx.Client", return_value=mock_client):
        df = load_dataframe(
            {
                "url": "https://docs.google.com/spreadsheets/d/SHEET123/edit#gid=77",
            }
        )
        called_url = mock_client.get.call_args.args[0]
        assert called_url == "https://docs.google.com/spreadsheets/d/SHEET123/export?format=csv&gid=77"
        assert len(df) == 2
        assert list(df["sku"]) == ["A1", "B2"]


def test_rest_api_connector_get_post_and_env_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTIMA_API_KEY", "super-secret-token")
    assert resolve_env_value("env:OPTIMA_API_KEY") == "super-secret-token"
    assert resolve_env_value("Bearer env:OPTIMA_API_KEY") == "Bearer super-secret-token"

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "status": "ok",
        "data": {
            "products": [
                {"sku": "R1", "pricing": {"cost": 12.0, "sell": 22.0}},
                {"sku": "R2", "pricing": {"cost": 7.0, "sell": 14.0}},
            ]
        },
    }
    mock_resp.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.return_value = mock_resp

    with patch("optima_mcp.connectors.web.httpx.Client", return_value=mock_client):
        df = load_dataframe(
            {
                "source_type": "rest_api",
                "url": "https://erp.example.com/api/v1/inventory/query",
                "method": "POST",
                "headers": {"Authorization": "Bearer env:OPTIMA_API_KEY"},
                "body": {"warehouse": "MAIN"},
                "data_path": "data.products",
            }
        )
        assert mock_client.post.call_args.kwargs["headers"]["Authorization"] == "Bearer super-secret-token"
        assert len(df) == 2
        assert "pricing.cost" in df.columns
        assert list(df["sku"]) == ["R1", "R2"]
