"""FastMCP server exposing Optima-MCP Tools, Resources, and Prompts over stdio."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    # mcp >= 2.0 renamed FastMCP to MCPServer
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # pragma: no cover - compatibility with mcp 1.x
    from mcp.server.fastmcp import FastMCP  # type: ignore[no-redef]

from optima_mcp.connectors.base import inspect_dataframe, load_dataframe
from optima_mcp.solvers.allocation import solve_allocation as _solve_allocation
from optima_mcp.solvers.procurement import solve_procurement as _solve_procurement
from optima_mcp.solvers.routing import solve_routing as _solve_routing
from optima_mcp.solvers.scenarios import run_what_if_scenarios as _run_what_if_scenarios
from optima_mcp.solvers.scheduling import solve_scheduling as _solve_scheduling
from optima_mcp.validation.schemas import dry_run_validate as _dry_run_validate

_FALLBACK_SKILL_MD = """---
name: optima-mcp
description: Universal zero-hallucination Operations Research & Decision Intelligence engine. Use whenever the user needs optimal purchasing, inventory planning, vehicle routing, shift scheduling, resource allocation, or what-if scenario math from files, SQL databases, Google Sheets, or REST APIs.
---

# STRICT EXECUTION PROTOCOL (ZERO HALLUCINATION)

1. **NEVER DO MENTAL MATH OR GUESS DECISIONS:** You are strictly forbidden from calculating optimal purchase quantities, routes, shift schedules, or budget allocations yourself. Your role is purely an intelligent data adapter and explainer.
2. **NEVER PULL RAW DATASETS INTO CONTEXT:** Always start with `inspect_source` to see column names, types, and at most 3 sample rows.
3. **MANDATORY DRY-RUN VALIDATION:** Before calling any `solve_*` tool, you MUST call `dry_run_validate`. Review all warnings (negative margins, unit mismatches like carton vs. single piece, budget/storage conflicts) and confirm key assumptions with the user before solving.
4. **FAITHFUL REPORTING:**
   - Report every number from `solve_*` exactly as returned by the mathematical solver — never alter or invent figures.
   - Always surface KPIs (budget utilization, storage utilization, solver name & solve time) and `skipped_opportunities` so the user understands *why* items were selected or skipped.
   - If the solver returns `Infeasible`, read the diagnostic report, explain the exact conflicting constraints, and offer to run `run_what_if_scenarios` to test relaxed options.
"""


def _load_skill_markdown() -> str:
    """Load SKILL.md from installed package data or repository root."""
    pkg_dir = Path(__file__).resolve().parent
    candidates = [
        pkg_dir / "SKILL.md",
        pkg_dir.parent.parent / "SKILL.md",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return _FALLBACK_SKILL_MD


mcp = FastMCP(
    "optima-mcp",
    instructions=(
        "Optima-MCP is a deterministic Operations Research engine. ALWAYS follow the strict 3-step "
        "zero-hallucination pipeline: (1) inspect_source -> (2) dry_run_validate -> (3) solve_*. "
        "Never calculate optimal numbers mentally and never dump full tables into LLM context."
    ),
)


# ---------------------------------------------------------------------------
# MCP Tools (Strict 3-Step Pipeline)
# ---------------------------------------------------------------------------


@mcp.tool()
def inspect_source(source_config: dict[str, Any]) -> dict[str, Any]:
    """STEP 1 OF 3 (MANDATORY FIRST CALL): Inspect any data source safely with zero context flooding.

    Loads the dataset locally into Pandas and returns ONLY schema metadata:
    row_count, columns, dtypes, null_counts, sample_rows (max 3), and numeric_summary (min/max/mean).
    NEVER returns full raw tables into the LLM context window.

    Supported source_config keys:
      - source_type: 'auto' (default), 'csv', 'excel', 'json', 'parquet', 'sql', 'google_sheets', 'rest_api'
      - Files: path, sheet_name (Excel), encoding, delimiter, data_path (JSON dot path)
      - SQL (strict read-only SELECT/WITH): connection_string (supports 'env:VAR_NAME'), query
      - Google Sheets: url (/edit URLs auto-converted to CSV export preserving gid)
      - REST API: url, method ('GET'|'POST'), headers (values support 'env:VAR_NAME'), params, body, data_path
    """
    df = load_dataframe(source_config)
    return inspect_dataframe(df, max_sample_rows=3)


@mcp.tool()
def dry_run_validate(
    module: str,
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """STEP 2 OF 3 (MANDATORY BEFORE ANY solve_* CALL): Validate schema, types, and business anomalies.

    Checks column existence, numeric coercibility, and strict Pydantic v2 models for the chosen
    module ('procurement', 'routing', 'scheduling', or 'allocation'). Emits explicit warnings for:
      - Negative costs/prices or negative-margin traps (cost >= selling_price)
      - Existing stock exceeding max_storage_space
      - Large unit-price variance (possible carton vs. single-unit mismatch)
      - Fleet capacity, staffing capacity, or minimum-allocation budget conflicts

    Review all warnings and confirm assumptions_summary with the user before calling Step 3 (solve_*).
    """
    return _dry_run_validate(
        module=module,
        source_config=source_config,
        mapping=mapping,
        constraints=constraints,
    )


@mcp.tool()
def solve_procurement(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """STEP 3 OF 3: Solve exact optimal Procurement & Inventory Replenishment via HiGHS MILP.

    PREREQUISITE: You MUST run `inspect_source` and `dry_run_validate(module='procurement', ...)` first.

    Required mapping keys: item_id, cost, selling_price, expected_demand, current_stock.
    Optional mapping keys: unit_space, min_order_qty, max_order_qty, shelf_life_days,
    supplier_id, supplier_fixed_shipping_cost.

    Constraints keys:
      - max_budget (required, > 0)
      - max_storage_space (optional)
      - min_demand_coverage (0.0 to 1.0, default 0.0), coverage_mode ('aggregate' | 'per_item')
      - planning_horizon_days (default 30.0)
      - objective: 'maximize_profit' (default) | 'minimize_cost' | 'maximize_roi'
      - supplier_fixed_shipping_costs: optional dict {supplier_id: fixed_cost}

    Returns exact deterministic purchase_plan, KPIs, skipped_opportunities with mathematical reasons,
    full audit_trail, and infeasibility diagnostics if Infeasible.
    """
    return _solve_procurement(
        source_config=source_config,
        mapping=mapping,
        constraints=constraints,
    )


@mcp.tool()
def solve_routing(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """STEP 3 OF 3: Solve exact Vehicle Routing Problem (CVRP) via Google OR-Tools Routing.

    PREREQUISITE: You MUST run `inspect_source` and `dry_run_validate(module='routing', ...)` first.

    Required mapping keys: stop_id, demand.
    Optional mapping keys: lat, lon (auto-computes Haversine km matrix), is_depot.

    Constraints keys:
      - num_vehicles, vehicle_capacities (list of positive numbers; auto-inferred from JSON if present)
      - max_route_distance (optional km cap per vehicle)
      - depot (optional dict {'stop_id', 'lat', 'lon'}), depot_index (default 0)
      - distance_matrix (optional explicit N x N matrix instead of lat/lon)
      - allow_drop_stops (bool, default False)

    Returns optimal stop sequences per vehicle, distances, load utilization %, dropped_stops, and audit_trail.
    """
    return _solve_routing(
        source_config=source_config,
        mapping=mapping,
        constraints=constraints,
    )


@mcp.tool()
def solve_scheduling(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """STEP 3 OF 3: Solve exact Shift & Workforce Scheduling via Google OR-Tools CP-SAT.

    PREREQUISITE: You MUST run `inspect_source` and `dry_run_validate(module='scheduling', ...)` first.

    Required mapping keys: staff_id, max_shifts.
    Optional mapping keys: name, skills (comma/semicolon/pipe-separated), shift_cost, unavailable_days.

    Constraints keys:
      - days (int or list of day names, default 7)
      - shifts (list of shift names, default ['Morning', 'Evening', 'Night'])
      - required_headcount (int, {shift: int}, or {day: {shift: int}})
      - required_skills (optional {skill: count} or {shift: {skill: count}})
      - no_back_to_back (bool, default True — enforces rest between consecutive shifts)
      - max_shifts_per_day (int, default 1)
      - objective: 'balance_workload' (default) | 'minimize_cost'

    Returns deterministic schedule, per-staff workload & utilization %, audit_trail, and diagnostics.
    """
    return _solve_scheduling(
        source_config=source_config,
        mapping=mapping,
        constraints=constraints,
    )


@mcp.tool()
def solve_allocation(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
) -> dict[str, Any]:
    """STEP 3 OF 3: Solve exact Resource, Capital, Shelf-Space, or Mix Allocation via HiGHS LP/MILP.

    PREREQUISITE: You MUST run `inspect_source` and `dry_run_validate(module='allocation', ...)` first.

    Required mapping keys: item_id, expected_return.
    Optional mapping keys: unit_cost (default 1.0), min_allocation, max_allocation, category.

    Constraints keys:
      - total_budget (or total_capacity, required > 0)
      - integer_vars (bool, default False — True solves MILP, False solves LP)
      - category_quotas (optional {category: {'min', 'max', 'min_pct', 'max_pct'}})
      - objective: 'maximize_return' (default) | 'minimize_cost'
      - min_total_return (optional lower bound on total return)

    Returns optimal allocations, skipped_opportunities, KPIs, audit_trail, and diagnostics.
    """
    return _solve_allocation(
        source_config=source_config,
        mapping=mapping,
        constraints=constraints,
    )


@mcp.tool()
def run_what_if_scenarios(
    module: str,
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    base_constraints: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run deterministic What-If sensitivity analysis across Baseline + up to 5 named scenarios.

    PREREQUISITE: Run `inspect_source` and `dry_run_validate` before solving scenarios.

    Each scenario dict supports:
      - name (required str)
      - description (optional str)
      - constraint_overrides (dict overriding keys in base_constraints, e.g. {'max_budget': 40000})
      - data_multipliers (optional dict scaling mapped numeric columns, e.g. {'cost': 1.08})

    Solves Baseline + every scenario deterministically and returns side-by-side KPIs & exact deltas.
    """
    return _run_what_if_scenarios(
        module=module,
        source_config=source_config,
        mapping=mapping,
        base_constraints=base_constraints,
        scenarios=scenarios,
    )


# ---------------------------------------------------------------------------
# MCP Resource & Prompt (Protocol Enforcement)
# ---------------------------------------------------------------------------


@mcp.resource("optima://protocol/strict-rules")
def strict_rules_resource() -> str:
    """Return the mandatory Zero-Hallucination Execution Protocol from SKILL.md."""
    return _load_skill_markdown()


@mcp.prompt()
def optima_strict_workflow(task_description: str = "") -> str:
    """Instruct any connected AI agent to strictly execute the Optima-MCP 3-step protocol."""
    rules = _load_skill_markdown()
    task_section = f"\n\n## User Task\n{task_description.strip()}" if task_description.strip() else ""
    return (
        "You are operating with the Optima-MCP deterministic Operations Research engine.\n"
        "You MUST obey every rule below without exception:\n\n"
        f"{rules}\n\n"
        "## Required Tool Sequence\n"
        "1. Call `inspect_source` to examine columns, types, and at most 3 sample rows.\n"
        "2. Map columns and call `dry_run_validate` — surface all warnings and confirm assumptions.\n"
        "3. Call the appropriate `solve_*` tool (and optionally `run_what_if_scenarios`) and report "
        "all numbers faithfully with KPIs, solver name, solve time, and skipped_opportunities."
        f"{task_section}"
    )


def main() -> None:
    """Entry point for the `optima-mcp` CLI over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
