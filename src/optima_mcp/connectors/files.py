"""Local file connector supporting CSV, Excel (.xlsx/.xls), JSON, and Parquet."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from optima_mcp.connectors.base import BaseConnector, DataSourceConfig


def extract_json_path(payload: Any, data_path: str | None) -> Any:
    """Navigate a dot-notation path inside a JSON object/array (e.g. 'data.items')."""
    if not data_path:
        return payload

    current = payload
    for segment in data_path.split("."):
        if isinstance(current, dict):
            if segment not in current:
                available = ", ".join(list(current.keys())[:10])
                raise KeyError(
                    f"JSON data_path segment '{segment}' not found. Available keys: [{available}]"
                )
            current = current[segment]
        elif isinstance(current, list):
            try:
                idx = int(segment)
                current = current[idx]
            except (ValueError, IndexError) as exc:
                raise KeyError(
                    f"Invalid list index '{segment}' in JSON data_path '{data_path}'."
                ) from exc
        else:
            raise KeyError(
                f"Cannot traverse segment '{segment}' on non-container type {type(current).__name__}."
            )
    return current


def json_payload_to_dataframe(payload: Any, data_path: str | None = None) -> pd.DataFrame:
    """Convert a parsed JSON payload into a flat Pandas DataFrame."""
    target = extract_json_path(payload, data_path)

    if isinstance(target, list):
        if len(target) == 0:
            return pd.DataFrame()
        if all(isinstance(item, dict) for item in target):
            return pd.json_normalize(target)
        return pd.DataFrame({"value": target})

    if isinstance(target, dict):
        if not target:
            return pd.DataFrame()

        # Column-oriented dict: {"col_a": [1, 2], "col_b": [3, 4]}
        values = list(target.values())
        if all(isinstance(v, list) for v in values):
            lengths = {len(v) for v in values}
            if len(lengths) == 1:
                return pd.DataFrame(target)

        # Common wrapper keys containing a list of records (e.g., {"stops": [...], "depot": {...}})
        preferred_keys = ("stops", "items", "records", "data", "rows", "orders", "results")
        for key in preferred_keys:
            if key in target and isinstance(target[key], list) and (
                len(target[key]) == 0 or isinstance(target[key][0], dict)
            ):
                return pd.json_normalize(target[key])

        # Single list-of-dicts key fallback
        list_dict_keys = [
            k
            for k, v in target.items()
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict)
        ]
        if len(list_dict_keys) == 1:
            return pd.json_normalize(target[list_dict_keys[0]])

        return pd.json_normalize([target])

    raise ValueError(
        f"Expected JSON list or object at data_path='{data_path}', got {type(target).__name__}."
    )


class FileConnector(BaseConnector):
    """Reads CSV, Excel (.xlsx/.xls), JSON, and Parquet files locally into Pandas."""

    def load(self, config: DataSourceConfig) -> pd.DataFrame:
        if not config.path:
            raise ValueError("FileConnector requires 'path' in source_config.")

        file_path = Path(config.path).expanduser()
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        if not file_path.is_file():
            raise ValueError(f"Path is not a regular file: {file_path}")

        resolved = config.resolved_source_type()

        if resolved == "csv":
            kwargs: dict[str, Any] = {}
            if config.encoding:
                kwargs["encoding"] = config.encoding
            if config.delimiter:
                kwargs["sep"] = config.delimiter
            elif file_path.suffix.lower() == ".tsv":
                kwargs["sep"] = "\t"
            return pd.read_csv(file_path, **kwargs)

        if resolved == "excel":
            sheet = 0 if config.sheet_name is None else config.sheet_name
            return pd.read_excel(file_path, sheet_name=sheet)

        if resolved == "json":
            raw_text = file_path.read_text(encoding=config.encoding or "utf-8")
            payload = json.loads(raw_text)
            return json_payload_to_dataframe(payload, data_path=config.data_path)

        if resolved == "parquet":
            return pd.read_parquet(file_path)

        raise ValueError(f"FileConnector cannot handle source_type '{resolved}'.")
