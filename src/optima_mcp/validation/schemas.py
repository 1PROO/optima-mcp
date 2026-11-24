"""Strict Pydantic v2 schemas and mandatory dry-run validation with anomaly detection."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import AliasChoices, BaseModel, Field, ValidationError, field_validator, model_validator

from optima_mcp.connectors.base import DataSourceConfig, load_dataframe

ModuleName = Literal["procurement", "routing", "scheduling", "allocation"]


# ---------------------------------------------------------------------------
# Shared Validation Output Models
# ---------------------------------------------------------------------------


class ValidationWarning(BaseModel):
    """Structured anomaly or business-rule warning raised during dry-run validation."""

    code: str
    severity: Literal["info", "warning", "critical"] = "warning"
    message: str
    affected_count: int = 0
    affected_items_sample: list[str] = Field(default_factory=list)


class DryRunReport(BaseModel):
    """Result of mandatory dry-run validation before calling any solver."""

    valid: bool
    ready_to_solve: bool
    module: str
    row_count: int
    valid_row_count: int
    errors: list[str] = Field(default_factory=list)
    warnings: list[ValidationWarning] = Field(default_factory=list)
    assumptions_summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Procurement Schemas
# ---------------------------------------------------------------------------


class ProcurementMapping(BaseModel):
    item_id: str
    cost: str
    selling_price: str
    expected_demand: str
    current_stock: str
    unit_space: str | None = None
    min_order_qty: str | None = None
    max_order_qty: str | None = None
    shelf_life_days: str | None = None
    supplier_id: str | None = None
    supplier_fixed_shipping_cost: str | None = None


class ProcurementConstraints(BaseModel):
    max_budget: float = Field(..., gt=0, description="Maximum cash spend including shipping.")
    max_storage_space: float | None = Field(
        default=None,
        ge=0,
        description="Optional maximum total post-order storage capacity.",
    )
    min_demand_coverage: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum fraction of total sellable demand that must be covered (0.0 to 1.0).",
    )
    coverage_mode: Literal["aggregate", "per_item"] = Field(
        default="aggregate",
        description="Apply min_demand_coverage across total demand ('aggregate') or every SKU ('per_item').",
    )
    planning_horizon_days: float = Field(
        default=30.0,
        gt=0,
        description="Planning horizon in days used when shelf_life_days is mapped.",
    )
    objective: Literal["maximize_profit", "minimize_cost", "maximize_roi"] = Field(
        default="maximize_profit"
    )
    supplier_fixed_shipping_costs: dict[str, float] | None = Field(
        default=None,
        description="Optional mapping of supplier_id -> fixed order shipping fee.",
    )


class ProcurementItem(BaseModel):
    item_id: str
    cost: float = Field(..., ge=0)
    selling_price: float = Field(..., ge=0)
    expected_demand: float = Field(..., ge=0)
    current_stock: float = Field(..., ge=0)
    unit_space: float = Field(default=0.0, ge=0)
    min_order_qty: int = Field(default=0, ge=0)
    max_order_qty: int | None = Field(default=None, ge=0)
    shelf_life_days: float | None = Field(default=None, gt=0)
    supplier_id: str | None = None
    supplier_fixed_shipping_cost: float = Field(default=0.0, ge=0)

    @model_validator(mode="after")
    def _check_moq_maxq(self) -> ProcurementItem:
        if self.max_order_qty is not None and self.min_order_qty > self.max_order_qty:
            raise ValueError(
                f"Item '{self.item_id}': min_order_qty ({self.min_order_qty}) "
                f"> max_order_qty ({self.max_order_qty})."
            )
        return self


# ---------------------------------------------------------------------------
# Routing Schemas
# ---------------------------------------------------------------------------


class RoutingMapping(BaseModel):
    stop_id: str
    demand: str
    lat: str | None = None
    lon: str | None = None
    is_depot: str | None = None


class RoutingConstraints(BaseModel):
    num_vehicles: int | None = Field(default=None, gt=0)
    vehicle_capacities: list[float] | None = None
    max_route_distance: float | None = Field(default=None, gt=0)
    depot: dict[str, Any] | None = None
    depot_index: int = Field(default=0, ge=0)
    distance_matrix: list[list[float]] | None = None
    allow_drop_stops: bool = False
    time_limit_seconds: int = Field(default=5, ge=1, le=60)

    @field_validator("vehicle_capacities")
    @classmethod
    def _positive_caps(cls, v: list[float] | None) -> list[float] | None:
        if v is not None:
            if len(v) == 0:
                raise ValueError("vehicle_capacities cannot be empty.")
            if any(cap <= 0 for cap in v):
                raise ValueError("All vehicle_capacities must be > 0.")
        return v


class RoutingStop(BaseModel):
    stop_id: str
    lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    demand: float = Field(default=0.0, ge=0)
    is_depot: bool = False


# ---------------------------------------------------------------------------
# Scheduling Schemas
# ---------------------------------------------------------------------------


class SchedulingMapping(BaseModel):
    staff_id: str
    max_shifts: str
    name: str | None = None
    skills: str | None = None
    shift_cost: str | None = None
    unavailable_days: str | None = None


class SchedulingConstraints(BaseModel):
    days: int | list[str] = 7
    shifts: list[str] = Field(default_factory=lambda: ["Morning", "Evening", "Night"])
    required_headcount: int | dict[str, int] | dict[str, dict[str, int]] = 1
    required_skills: dict[str, int] | dict[str, dict[str, int]] | None = None
    no_back_to_back: bool = True
    max_shifts_per_day: int = Field(default=1, ge=1)
    objective: Literal["balance_workload", "minimize_cost"] = "balance_workload"
    time_limit_seconds: int = Field(default=10, ge=1, le=60)

    @field_validator("days")
    @classmethod
    def _validate_days(cls, v: int | list[str]) -> int | list[str]:
        if isinstance(v, int) and v <= 0:
            raise ValueError("days must be > 0.")
        if isinstance(v, list) and len(v) == 0:
            raise ValueError("days list must not be empty.")
        return v

    @field_validator("shifts")
    @classmethod
    def _validate_shifts(cls, v: list[str]) -> list[str]:
        if len(v) == 0:
            raise ValueError("shifts list must not be empty.")
        return v

    def day_names(self) -> list[str]:
        if isinstance(self.days, int):
            return [f"Day_{i + 1}" for i in range(self.days)]
        return [str(d) for d in self.days]


class StaffMember(BaseModel):
    staff_id: str
    name: str | None = None
    skills: list[str] = Field(default_factory=list)
    max_shifts: int = Field(..., ge=0)
    shift_cost: float = Field(default=0.0, ge=0)
    unavailable_days: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Allocation Schemas
# ---------------------------------------------------------------------------


class AllocationMapping(BaseModel):
    item_id: str
    expected_return: str
    unit_cost: str | None = None
    min_allocation: str | None = None
    max_allocation: str | None = None
    category: str | None = None


class AllocationConstraints(BaseModel):
    total_budget: float = Field(
        ...,
        gt=0,
        validation_alias=AliasChoices("total_budget", "total_capacity"),
    )
    integer_vars: bool = False
    category_quotas: dict[str, dict[str, float]] | None = None
    objective: Literal["maximize_return", "minimize_cost"] = "maximize_return"
    min_total_return: float | None = None


class AllocationItem(BaseModel):
    item_id: str
    expected_return: float
    unit_cost: float = Field(default=1.0, gt=0)
    min_allocation: float = Field(default=0.0, ge=0)
    max_allocation: float | None = Field(default=None, ge=0)
    category: str | None = None

    @model_validator(mode="after")
    def _check_bounds(self) -> AllocationItem:
        if self.max_allocation is not None and self.min_allocation > self.max_allocation:
            raise ValueError(
                f"Item '{self.item_id}': min_allocation ({self.min_allocation}) "
                f"> max_allocation ({self.max_allocation})."
            )
        return self


# ---------------------------------------------------------------------------
# What-If Scenario Schema
# ---------------------------------------------------------------------------


class ScenarioSpec(BaseModel):
    name: str = Field(..., min_length=1)
    description: str | None = None
    constraint_overrides: dict[str, Any] = Field(default_factory=dict)
    data_multipliers: dict[str, float] | None = Field(
        default=None,
        description="Optional multiplicative factors on mapped numeric columns (e.g. {'cost': 1.08}).",
    )


# ---------------------------------------------------------------------------
# DataFrame Parsing Helpers (Strict Coercion & Error Collection)
# ---------------------------------------------------------------------------


def _check_columns_exist(df: pd.DataFrame, mapping: BaseModel) -> list[str]:
    missing: list[str] = []
    df_cols = set(df.columns.astype(str))
    for field_name, col_name in mapping.model_dump().items():
        if col_name is not None and col_name not in df_cols:
            missing.append(
                f"Mapped column '{col_name}' for field '{field_name}' does not exist. "
                f"Available columns: {sorted(df_cols)}"
            )
    return missing


def _coerce_float(val: Any, default: float | None = None) -> tuple[float | None, bool]:
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default, default is not None
    try:
        if pd.isna(val):
            return default, default is not None
    except (TypeError, ValueError):
        pass
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None, False
        return f, True
    except (TypeError, ValueError):
        return None, False


def _coerce_int(val: Any, default: int | None = None) -> tuple[int | None, bool]:
    fval, ok = _coerce_float(val, float(default) if default is not None else None)
    if not ok or fval is None:
        return None, False
    return int(round(fval)), True


def _parse_token_list(val: Any) -> list[str]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return [str(x).strip() for x in val if str(x).strip()]
    try:
        if pd.isna(val):
            return []
    except (TypeError, ValueError):
        pass
    text = str(val).strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return []
    for sep in (";", "|", ","):
        if sep in text:
            return [p.strip() for p in text.split(sep) if p.strip()]
    return [text]


def _maybe_load_routing_json_metadata(source_config: dict[str, Any] | DataSourceConfig) -> dict[str, Any]:
    """If source is a structured JSON file containing top-level depot/vehicles, read those defaults."""
    cfg = (
        source_config
        if isinstance(source_config, DataSourceConfig)
        else DataSourceConfig.model_validate(source_config)
    )
    if cfg.path and Path(cfg.path).suffix.lower() == ".json":
        file_path = Path(cfg.path).expanduser()
        if file_path.is_file():
            try:
                raw = json.loads(file_path.read_text(encoding=cfg.encoding or "utf-8"))
                if isinstance(raw, dict):
                    return raw
            except Exception:
                return {}
    return {}


def parse_procurement_dataframe(
    df: pd.DataFrame,
    mapping: ProcurementMapping,
    constraints: ProcurementConstraints,
) -> tuple[list[ProcurementItem], list[str]]:
    errors = _check_columns_exist(df, mapping)
    if errors:
        return [], errors

    items: list[ProcurementItem] = []
    seen_ids: set[str] = set()
    shipping_map = constraints.supplier_fixed_shipping_costs or {}

    for idx, row in df.iterrows():
        item_id = str(row[mapping.item_id]).strip()
        if not item_id or item_id.lower() == "nan":
            errors.append(f"Row {idx}: missing item_id.")
            continue
        if item_id in seen_ids:
            errors.append(f"Row {idx}: duplicate item_id '{item_id}'.")
            continue
        seen_ids.add(item_id)

        cost, ok_cost = _coerce_float(row[mapping.cost])
        price, ok_price = _coerce_float(row[mapping.selling_price])
        demand, ok_demand = _coerce_float(row[mapping.expected_demand])
        stock, ok_stock = _coerce_float(row[mapping.current_stock], default=0.0)

        if not all([ok_cost, ok_price, ok_demand, ok_stock]):
            errors.append(
                f"Row {idx} ('{item_id}'): non-numeric or missing required value "
                f"(cost={row[mapping.cost]}, selling_price={row[mapping.selling_price]}, "
                f"expected_demand={row[mapping.expected_demand]}, current_stock={row[mapping.current_stock]})."
            )
            continue

        unit_space = 1.0 if constraints.max_storage_space is not None else 0.0
        if mapping.unit_space:
            us, ok_us = _coerce_float(row[mapping.unit_space], default=0.0)
            if not ok_us or us is None:
                errors.append(f"Row {idx} ('{item_id}'): invalid unit_space '{row[mapping.unit_space]}'.")
                continue
            unit_space = us

        moq = 0
        if mapping.min_order_qty:
            m_val, ok_m = _coerce_int(row[mapping.min_order_qty], default=0)
            if not ok_m or m_val is None:
                errors.append(f"Row {idx} ('{item_id}'): invalid min_order_qty '{row[mapping.min_order_qty]}'.")
                continue
            moq = m_val

        maxq: int | None = None
        if mapping.max_order_qty:
            raw_maxq = row[mapping.max_order_qty]
            if raw_maxq is not None and not (isinstance(raw_maxq, float) and math.isnan(raw_maxq)):
                try:
                    is_na = pd.isna(raw_maxq)
                except (TypeError, ValueError):
                    is_na = False
                if not is_na and str(raw_maxq).strip() != "":
                    mx_val, ok_mx = _coerce_int(raw_maxq)
                    if not ok_mx:
                        errors.append(f"Row {idx} ('{item_id}'): invalid max_order_qty '{raw_maxq}'.")
                        continue
                    maxq = mx_val

        shelf_life: float | None = None
        if mapping.shelf_life_days:
            raw_sl = row[mapping.shelf_life_days]
            if raw_sl is not None and not (isinstance(raw_sl, float) and math.isnan(raw_sl)):
                try:
                    is_na = pd.isna(raw_sl)
                except (TypeError, ValueError):
                    is_na = False
                if not is_na and str(raw_sl).strip() != "":
                    sl_val, ok_sl = _coerce_float(raw_sl)
                    if not ok_sl:
                        errors.append(f"Row {idx} ('{item_id}'): invalid shelf_life_days '{raw_sl}'.")
                        continue
                    shelf_life = sl_val

        supplier_id: str | None = None
        if mapping.supplier_id:
            raw_sup = row[mapping.supplier_id]
            if raw_sup is not None and not pd.isna(raw_sup) and str(raw_sup).strip():
                supplier_id = str(raw_sup).strip()

        fixed_ship = 0.0
        if mapping.supplier_fixed_shipping_cost:
            fs_val, ok_fs = _coerce_float(row[mapping.supplier_fixed_shipping_cost], default=0.0)
            if not ok_fs or fs_val is None:
                errors.append(
                    f"Row {idx} ('{item_id}'): invalid supplier_fixed_shipping_cost "
                    f"'{row[mapping.supplier_fixed_shipping_cost]}'."
                )
                continue
            fixed_ship = fs_val
        elif supplier_id and supplier_id in shipping_map:
            fixed_ship = float(shipping_map[supplier_id])

        try:
            items.append(
                ProcurementItem(
                    item_id=item_id,
                    cost=float(cost),  # type: ignore[arg-type]
                    selling_price=float(price),  # type: ignore[arg-type]
                    expected_demand=float(demand),  # type: ignore[arg-type]
                    current_stock=float(stock),  # type: ignore[arg-type]
                    unit_space=unit_space,
                    min_order_qty=moq,
                    max_order_qty=maxq,
                    shelf_life_days=shelf_life,
                    supplier_id=supplier_id,
                    supplier_fixed_shipping_cost=fixed_ship,
                )
            )
        except ValidationError as exc:
            first_err = exc.errors()[0]["msg"]
            errors.append(f"Row {idx} ('{item_id}'): {first_err}")

    return items, errors


def parse_routing_dataframe(
    df: pd.DataFrame,
    mapping: RoutingMapping,
    constraints: RoutingConstraints,
    source_config: dict[str, Any] | DataSourceConfig | None = None,
) -> tuple[list[RoutingStop], RoutingConstraints, list[str]]:
    errors = _check_columns_exist(df, mapping)
    if errors:
        return [], constraints, errors

    # Merge optional JSON metadata (depot / vehicles) if not explicitly provided
    resolved_constraints = constraints.model_copy(deep=True)
    if source_config is not None:
        meta = _maybe_load_routing_json_metadata(source_config)
        if resolved_constraints.depot is None and isinstance(meta.get("depot"), dict):
            resolved_constraints.depot = meta["depot"]
        if isinstance(meta.get("vehicles"), list) and meta["vehicles"]:
            if resolved_constraints.vehicle_capacities is None:
                caps = [
                    float(v["capacity"])
                    for v in meta["vehicles"]
                    if isinstance(v, dict) and "capacity" in v
                ]
                if caps:
                    resolved_constraints.vehicle_capacities = caps
            if resolved_constraints.num_vehicles is None:
                resolved_constraints.num_vehicles = len(meta["vehicles"])

    if resolved_constraints.num_vehicles is None and resolved_constraints.vehicle_capacities:
        resolved_constraints.num_vehicles = len(resolved_constraints.vehicle_capacities)
    if resolved_constraints.num_vehicles is None:
        errors.append("Routing constraints require 'num_vehicles' or 'vehicle_capacities'.")
    if resolved_constraints.vehicle_capacities is None:
        errors.append("Routing constraints require 'vehicle_capacities'.")
    elif (
        resolved_constraints.num_vehicles is not None
        and len(resolved_constraints.vehicle_capacities) == 1
        and resolved_constraints.num_vehicles > 1
    ):
        resolved_constraints.vehicle_capacities = (
            resolved_constraints.vehicle_capacities * resolved_constraints.num_vehicles
        )
    elif (
        resolved_constraints.num_vehicles is not None
        and len(resolved_constraints.vehicle_capacities) != resolved_constraints.num_vehicles
    ):
        errors.append(
            f"Length of vehicle_capacities ({len(resolved_constraints.vehicle_capacities)}) "
            f"must match num_vehicles ({resolved_constraints.num_vehicles})."
        )

    stops: list[RoutingStop] = []
    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        stop_id = str(row[mapping.stop_id]).strip()
        if not stop_id or stop_id.lower() == "nan":
            errors.append(f"Row {idx}: missing stop_id.")
            continue
        if stop_id in seen_ids:
            errors.append(f"Row {idx}: duplicate stop_id '{stop_id}'.")
            continue
        seen_ids.add(stop_id)

        demand, ok_d = _coerce_float(row[mapping.demand], default=0.0)
        if not ok_d or demand is None:
            errors.append(f"Row {idx} ('{stop_id}'): invalid demand '{row[mapping.demand]}'.")
            continue

        lat: float | None = None
        lon: float | None = None
        if mapping.lat and mapping.lon:
            lat, ok_lat = _coerce_float(row[mapping.lat])
            lon, ok_lon = _coerce_float(row[mapping.lon])
            if not (ok_lat and ok_lon):
                errors.append(
                    f"Row {idx} ('{stop_id}'): invalid coordinates lat={row[mapping.lat]}, lon={row[mapping.lon]}."
                )
                continue

        is_depot = False
        if mapping.is_depot:
            raw_flag = row[mapping.is_depot]
            if isinstance(raw_flag, bool):
                is_depot = raw_flag
            else:
                is_depot = str(raw_flag).strip().lower() in {"true", "1", "yes", "depot"}

        try:
            stops.append(
                RoutingStop(
                    stop_id=stop_id,
                    lat=lat,
                    lon=lon,
                    demand=demand,
                    is_depot=is_depot,
                )
            )
        except ValidationError as exc:
            errors.append(f"Row {idx} ('{stop_id}'): {exc.errors()[0]['msg']}")

    if resolved_constraints.distance_matrix is None and (not mapping.lat or not mapping.lon):
        errors.append(
            "Routing requires either lat/lon columns in mapping or an explicit 'distance_matrix' in constraints."
        )

    return stops, resolved_constraints, errors


def parse_scheduling_dataframe(
    df: pd.DataFrame,
    mapping: SchedulingMapping,
) -> tuple[list[StaffMember], list[str]]:
    errors = _check_columns_exist(df, mapping)
    if errors:
        return [], errors

    staff: list[StaffMember] = []
    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        staff_id = str(row[mapping.staff_id]).strip()
        if not staff_id or staff_id.lower() == "nan":
            errors.append(f"Row {idx}: missing staff_id.")
            continue
        if staff_id in seen_ids:
            errors.append(f"Row {idx}: duplicate staff_id '{staff_id}'.")
            continue
        seen_ids.add(staff_id)

        max_shifts, ok_ms = _coerce_int(row[mapping.max_shifts])
        if not ok_ms or max_shifts is None:
            errors.append(f"Row {idx} ('{staff_id}'): invalid max_shifts '{row[mapping.max_shifts]}'.")
            continue

        name: str | None = None
        if mapping.name and not pd.isna(row[mapping.name]):
            name = str(row[mapping.name]).strip()

        skills = _parse_token_list(row[mapping.skills]) if mapping.skills else []
        unavail = _parse_token_list(row[mapping.unavailable_days]) if mapping.unavailable_days else []

        shift_cost = 0.0
        if mapping.shift_cost:
            sc, ok_sc = _coerce_float(row[mapping.shift_cost], default=0.0)
            if not ok_sc or sc is None:
                errors.append(f"Row {idx} ('{staff_id}'): invalid shift_cost '{row[mapping.shift_cost]}'.")
                continue
            shift_cost = sc

        try:
            staff.append(
                StaffMember(
                    staff_id=staff_id,
                    name=name,
                    skills=skills,
                    max_shifts=max_shifts,
                    shift_cost=shift_cost,
                    unavailable_days=unavail,
                )
            )
        except ValidationError as exc:
            errors.append(f"Row {idx} ('{staff_id}'): {exc.errors()[0]['msg']}")

    return staff, errors


def parse_allocation_dataframe(
    df: pd.DataFrame,
    mapping: AllocationMapping,
) -> tuple[list[AllocationItem], list[str]]:
    errors = _check_columns_exist(df, mapping)
    if errors:
        return [], errors

    items: list[AllocationItem] = []
    seen_ids: set[str] = set()

    for idx, row in df.iterrows():
        item_id = str(row[mapping.item_id]).strip()
        if not item_id or item_id.lower() == "nan":
            errors.append(f"Row {idx}: missing item_id.")
            continue
        if item_id in seen_ids:
            errors.append(f"Row {idx}: duplicate item_id '{item_id}'.")
            continue
        seen_ids.add(item_id)

        exp_ret, ok_ret = _coerce_float(row[mapping.expected_return])
        if not ok_ret or exp_ret is None:
            errors.append(
                f"Row {idx} ('{item_id}'): invalid expected_return '{row[mapping.expected_return]}'."
            )
            continue

        unit_cost = 1.0
        if mapping.unit_cost:
            uc, ok_uc = _coerce_float(row[mapping.unit_cost])
            if not ok_uc or uc is None:
                errors.append(f"Row {idx} ('{item_id}'): invalid unit_cost '{row[mapping.unit_cost]}'.")
                continue
            unit_cost = uc

        min_alloc = 0.0
        if mapping.min_allocation:
            mn, ok_mn = _coerce_float(row[mapping.min_allocation], default=0.0)
            if not ok_mn or mn is None:
                errors.append(
                    f"Row {idx} ('{item_id}'): invalid min_allocation '{row[mapping.min_allocation]}'."
                )
                continue
            min_alloc = mn

        max_alloc: float | None = None
        if mapping.max_allocation:
            raw_mx = row[mapping.max_allocation]
            if raw_mx is not None and not (isinstance(raw_mx, float) and math.isnan(raw_mx)):
                try:
                    is_na = pd.isna(raw_mx)
                except (TypeError, ValueError):
                    is_na = False
                if not is_na and str(raw_mx).strip() != "":
                    mx, ok_mx = _coerce_float(raw_mx)
                    if not ok_mx:
                        errors.append(f"Row {idx} ('{item_id}'): invalid max_allocation '{raw_mx}'.")
                        continue
                    max_alloc = mx

        category: str | None = None
        if mapping.category and not pd.isna(row[mapping.category]):
            cat = str(row[mapping.category]).strip()
            if cat:
                category = cat

        try:
            items.append(
                AllocationItem(
                    item_id=item_id,
                    expected_return=exp_ret,
                    unit_cost=unit_cost,
                    min_allocation=min_alloc,
                    max_allocation=max_alloc,
                    category=category,
                )
            )
        except ValidationError as exc:
            errors.append(f"Row {idx} ('{item_id}'): {exc.errors()[0]['msg']}")

    return items, errors


# ---------------------------------------------------------------------------
# Mandatory Dry-Run Validator
# ---------------------------------------------------------------------------


def dry_run_validate(
    module: str,
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """Validate source schema, numeric types, Pydantic models, and domain anomalies.

    Emits explicit warnings for negative margins, unit mismatches, storage overflows,
    budget/capacity conflicts, and returns a structured assumptions summary for confirmation.
    """
    valid_modules = {"procurement", "routing", "scheduling", "allocation"}
    if module not in valid_modules:
        return DryRunReport(
            valid=False,
            ready_to_solve=False,
            module=module,
            row_count=0,
            valid_row_count=0,
            errors=[f"Unsupported module '{module}'. Choose from: {sorted(valid_modules)}"],
        ).model_dump()

    try:
        df = load_dataframe(source_config)
    except Exception as exc:
        return DryRunReport(
            valid=False,
            ready_to_solve=False,
            module=module,
            row_count=0,
            valid_row_count=0,
            errors=[f"Failed to load data source: {exc}"],
        ).model_dump()

    row_count = int(len(df))
    if row_count == 0:
        return DryRunReport(
            valid=False,
            ready_to_solve=False,
            module=module,
            row_count=0,
            valid_row_count=0,
            errors=["Loaded dataset is empty (0 rows)."],
        ).model_dump()

    errors: list[str] = []
    warnings: list[ValidationWarning] = []
    assumptions: dict[str, Any] = {"module": module, "row_count": row_count, "mapping": mapping}
    valid_row_count = 0

    if module == "procurement":
        try:
            p_map = ProcurementMapping.model_validate(mapping)
            p_cons = ProcurementConstraints.model_validate(constraints)
        except ValidationError as exc:
            errors.extend([f"{e['loc']}: {e['msg']}" for e in exc.errors()])
            return DryRunReport(
                valid=False,
                ready_to_solve=False,
                module=module,
                row_count=row_count,
                valid_row_count=0,
                errors=errors,
            ).model_dump()

        items, parse_errors = parse_procurement_dataframe(df, p_map, p_cons)
        errors.extend(parse_errors)
        valid_row_count = len(items)

        if items:
            # 1. Negative / zero margin trap check
            neg_margin = [it.item_id for it in items if it.cost >= it.selling_price]
            if neg_margin:
                warnings.append(
                    ValidationWarning(
                        code="NEGATIVE_MARGIN_TRAP",
                        severity="warning",
                        message=(
                            f"{len(neg_margin)} item(s) have cost >= selling_price (zero or negative unit margin). "
                            "Under maximize_profit, the solver will skip these unless forced by min_demand_coverage."
                        ),
                        affected_count=len(neg_margin),
                        affected_items_sample=neg_margin[:5],
                    )
                )

            # 2. Current stock exceeds max_storage_space
            current_space_used = sum(it.current_stock * it.unit_space for it in items)
            if p_cons.max_storage_space is not None and current_space_used > p_cons.max_storage_space:
                warnings.append(
                    ValidationWarning(
                        code="STORAGE_ALREADY_EXCEEDED",
                        severity="critical",
                        message=(
                            f"Existing current_stock already occupies {current_space_used:,.2f} storage units, "
                            f"which exceeds max_storage_space ({p_cons.max_storage_space:,.2f}). "
                            "Solving with this storage cap will be Infeasible."
                        ),
                        affected_count=len(items),
                    )
                )

            # 3. Large unit-price/cost variance (possible carton vs single-unit mismatch)
            positive_costs = [it.cost for it in items if it.cost > 0]
            if len(positive_costs) >= 3:
                med_cost = statistics.median(positive_costs)
                max_cost = max(positive_costs)
                min_cost = min(positive_costs)
                if med_cost > 0 and (max_cost / med_cost >= 25.0 or (min_cost > 0 and max_cost / min_cost >= 100.0)):
                    outliers = [it.item_id for it in items if it.cost >= med_cost * 25.0]
                    warnings.append(
                        ValidationWarning(
                            code="UNIT_PRICE_VARIANCE_HIGH",
                            severity="warning",
                            message=(
                                f"Unusually wide unit cost spread detected (min={min_cost:,.2f}, "
                                f"median={med_cost:,.2f}, max={max_cost:,.2f}). Verify there is no "
                                "carton/barrel vs. single-unit mismatch in cost, demand, or unit_space."
                            ),
                            affected_count=len(outliers),
                            affected_items_sample=outliers[:5],
                        )
                    )

            # 4. MOQ unaffordable under max_budget
            unaffordable_moq = [
                it.item_id for it in items if it.min_order_qty > 0 and it.min_order_qty * it.cost > p_cons.max_budget
            ]
            if unaffordable_moq:
                warnings.append(
                    ValidationWarning(
                        code="MOQ_EXCEEDS_BUDGET",
                        severity="warning",
                        message=(
                            f"{len(unaffordable_moq)} item(s) have minimum order cost (MOQ * cost) "
                            f"greater than max_budget ({p_cons.max_budget:,.2f}) and cannot be ordered."
                        ),
                        affected_count=len(unaffordable_moq),
                        affected_items_sample=unaffordable_moq[:5],
                    )
                )

            # 5. Summary assumptions for user confirmation
            total_demand_units = sum(it.expected_demand for it in items)
            total_stock_units = sum(it.current_stock for it in items)
            full_replenish_cost = sum(
                max(0.0, it.expected_demand - it.current_stock) * it.cost for it in items
            )
            assumptions.update(
                {
                    "constraints": p_cons.model_dump(),
                    "sku_count": len(items),
                    "total_expected_demand_units": round(total_demand_units, 2),
                    "total_current_stock_units": round(total_stock_units, 2),
                    "current_storage_used": round(current_space_used, 2),
                    "estimated_full_demand_replenishment_cost": round(full_replenish_cost, 2),
                    "suppliers_detected": len({it.supplier_id for it in items if it.supplier_id}),
                }
            )

    elif module == "routing":
        try:
            r_map = RoutingMapping.model_validate(mapping)
            r_cons = RoutingConstraints.model_validate(constraints)
        except ValidationError as exc:
            errors.extend([f"{e['loc']}: {e['msg']}" for e in exc.errors()])
            return DryRunReport(
                valid=False,
                ready_to_solve=False,
                module=module,
                row_count=row_count,
                valid_row_count=0,
                errors=errors,
            ).model_dump()

        stops, resolved_cons, parse_errors = parse_routing_dataframe(
            df, r_map, r_cons, source_config=source_config
        )
        errors.extend(parse_errors)
        valid_row_count = len(stops)

        if stops and resolved_cons.vehicle_capacities:
            total_demand = sum(s.demand for s in stops if not s.is_depot)
            total_cap = sum(resolved_cons.vehicle_capacities)
            max_cap = max(resolved_cons.vehicle_capacities)

            if total_demand > total_cap:
                warnings.append(
                    ValidationWarning(
                        code="FLEET_CAPACITY_INSUFFICIENT",
                        severity="critical" if not resolved_cons.allow_drop_stops else "warning",
                        message=(
                            f"Total stop demand ({total_demand:,.2f}) exceeds total fleet capacity "
                            f"({total_cap:,.2f}). Set allow_drop_stops=True or increase fleet capacity."
                        ),
                        affected_count=len(stops),
                    )
                )

            oversized = [s.stop_id for s in stops if not s.is_depot and s.demand > max_cap]
            if oversized:
                warnings.append(
                    ValidationWarning(
                        code="STOP_EXCEEDS_MAX_VEHICLE_CAPACITY",
                        severity="critical" if not resolved_cons.allow_drop_stops else "warning",
                        message=(
                            f"{len(oversized)} stop(s) have demand exceeding the largest vehicle capacity ({max_cap:,.2f})."
                        ),
                        affected_count=len(oversized),
                        affected_items_sample=oversized[:5],
                    )
                )

            assumptions.update(
                {
                    "constraints": resolved_cons.model_dump(exclude={"distance_matrix"}),
                    "stop_count": len(stops),
                    "total_demand": round(total_demand, 2),
                    "total_fleet_capacity": round(total_cap, 2),
                    "distance_mode": "explicit_matrix" if resolved_cons.distance_matrix else "haversine_km",
                }
            )

    elif module == "scheduling":
        try:
            s_map = SchedulingMapping.model_validate(mapping)
            s_cons = SchedulingConstraints.model_validate(constraints)
        except ValidationError as exc:
            errors.extend([f"{e['loc']}: {e['msg']}" for e in exc.errors()])
            return DryRunReport(
                valid=False,
                ready_to_solve=False,
                module=module,
                row_count=row_count,
                valid_row_count=0,
                errors=errors,
            ).model_dump()

        staff, parse_errors = parse_scheduling_dataframe(df, s_map)
        errors.extend(parse_errors)
        valid_row_count = len(staff)

        if staff:
            days = s_cons.day_names()
            total_required_slots = 0
            for d in days:
                for sh in s_cons.shifts:
                    if isinstance(s_cons.required_headcount, int):
                        req = s_cons.required_headcount
                    elif d in s_cons.required_headcount and isinstance(s_cons.required_headcount[d], dict):
                        req = int(s_cons.required_headcount[d].get(sh, 0))  # type: ignore[union-attr]
                    else:
                        req = int(s_cons.required_headcount.get(sh, 0))  # type: ignore[arg-type]
                    total_required_slots += req

            total_staff_capacity = sum(st.max_shifts for st in staff)
            if total_required_slots > total_staff_capacity:
                warnings.append(
                    ValidationWarning(
                        code="INSUFFICIENT_STAFF_CAPACITY",
                        severity="critical",
                        message=(
                            f"Total required shift slots ({total_required_slots}) exceeds total available "
                            f"staff shift capacity ({total_staff_capacity}). Schedule will be Infeasible."
                        ),
                        affected_count=len(staff),
                    )
                )

            zero_cap = [st.staff_id for st in staff if st.max_shifts == 0]
            if zero_cap:
                warnings.append(
                    ValidationWarning(
                        code="ZERO_MAX_SHIFTS_STAFF",
                        severity="info",
                        message=f"{len(zero_cap)} staff member(s) have max_shifts=0 and will not be scheduled.",
                        affected_count=len(zero_cap),
                        affected_items_sample=zero_cap[:5],
                    )
                )

            assumptions.update(
                {
                    "constraints": s_cons.model_dump(),
                    "staff_count": len(staff),
                    "num_days": len(days),
                    "num_shifts_per_day": len(s_cons.shifts),
                    "total_required_shift_slots": total_required_slots,
                    "total_available_staff_shifts": total_staff_capacity,
                }
            )

    elif module == "allocation":
        try:
            a_map = AllocationMapping.model_validate(mapping)
            a_cons = AllocationConstraints.model_validate(constraints)
        except ValidationError as exc:
            errors.extend([f"{e['loc']}: {e['msg']}" for e in exc.errors()])
            return DryRunReport(
                valid=False,
                ready_to_solve=False,
                module=module,
                row_count=row_count,
                valid_row_count=0,
                errors=errors,
            ).model_dump()

        items_alloc, parse_errors = parse_allocation_dataframe(df, a_map)
        errors.extend(parse_errors)
        valid_row_count = len(items_alloc)

        if items_alloc:
            min_required_budget = sum(it.min_allocation * it.unit_cost for it in items_alloc)
            if min_required_budget > a_cons.total_budget:
                warnings.append(
                    ValidationWarning(
                        code="MIN_BOUNDS_EXCEED_BUDGET",
                        severity="critical",
                        message=(
                            f"Sum of mandatory min_allocation spend ({min_required_budget:,.2f}) exceeds "
                            f"total_budget ({a_cons.total_budget:,.2f}). Allocation will be Infeasible."
                        ),
                        affected_count=len(items_alloc),
                    )
                )

            neg_ret = [it.item_id for it in items_alloc if it.expected_return <= 0]
            if neg_ret:
                warnings.append(
                    ValidationWarning(
                        code="NON_POSITIVE_RETURN_ITEMS",
                        severity="warning",
                        message=(
                            f"{len(neg_ret)} item(s) have expected_return <= 0 and will receive only "
                            "their min_allocation under maximize_return."
                        ),
                        affected_count=len(neg_ret),
                        affected_items_sample=neg_ret[:5],
                    )
                )

            assumptions.update(
                {
                    "constraints": a_cons.model_dump(),
                    "item_count": len(items_alloc),
                    "min_mandatory_budget": round(min_required_budget, 2),
                    "categories_detected": sorted({it.category for it in items_alloc if it.category}),
                }
            )

    is_valid = len(errors) == 0 and valid_row_count > 0
    return DryRunReport(
        valid=is_valid,
        ready_to_solve=is_valid,
        module=module,
        row_count=row_count,
        valid_row_count=valid_row_count,
        errors=errors[:25],
        warnings=warnings,
        assumptions_summary=assumptions,
    ).model_dump()
