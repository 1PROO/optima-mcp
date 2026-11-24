"""What-If Sensitivity & Multi-Scenario Comparison Engine."""

from __future__ import annotations

import copy
import time
from typing import Any

import pandas as pd

from optima_mcp.connectors.base import load_dataframe
from optima_mcp.solvers.allocation import solve_allocation
from optima_mcp.solvers.procurement import solve_procurement
from optima_mcp.solvers.routing import solve_routing
from optima_mcp.solvers.scheduling import solve_scheduling
from optima_mcp.validation.audit import build_audit_trail
from optima_mcp.validation.schemas import ScenarioSpec

MAX_SCENARIOS = 5

_PRIMARY_KPIS: dict[str, list[str]] = {
    "procurement": [
        "total_spend",
        "expected_profit",
        "roi_pct",
        "budget_utilization_pct",
        "demand_coverage_pct",
        "items_ordered_count",
    ],
    "routing": [
        "total_distance_km",
        "vehicles_used",
        "stops_served",
        "stops_dropped",
        "fleet_load_utilization_pct",
    ],
    "scheduling": [
        "total_shifts_filled",
        "staff_utilized",
        "workload_spread",
        "total_labor_cost",
    ],
    "allocation": [
        "total_allocated",
        "total_expected_return",
        "weighted_return_pct",
        "budget_utilization_pct",
        "items_funded_count",
    ],
}


def _apply_data_multipliers(
    df: pd.DataFrame,
    mapping: dict[str, Any],
    multipliers: dict[str, float] | None,
) -> pd.DataFrame:
    if not multipliers:
        return df
    df_copy = df.copy()
    for key, factor in multipliers.items():
        # Allow referencing either the logical mapped field (e.g. 'cost') or raw column name
        col_name = mapping.get(key, key)
        if col_name not in df_copy.columns:
            raise ValueError(
                f"data_multipliers target '{key}' (resolved column '{col_name}') not found in dataset."
            )
        df_copy[col_name] = pd.to_numeric(df_copy[col_name], errors="raise") * float(factor)
    return df_copy


def _dispatch_solver(
    module: str,
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    if module == "procurement":
        return solve_procurement(source_config, mapping, constraints, df_override=df)
    if module == "routing":
        return solve_routing(source_config, mapping, constraints, df_override=df)
    if module == "scheduling":
        return solve_scheduling(source_config, mapping, constraints, df_override=df)
    if module == "allocation":
        return solve_allocation(source_config, mapping, constraints, df_override=df)
    raise ValueError(
        f"Unsupported module '{module}'. Choose from: ['allocation', 'procurement', 'routing', 'scheduling']."
    )


def _compute_kpi_deltas(
    module: str,
    baseline_summary: dict[str, Any],
    scenario_summary: dict[str, Any],
) -> dict[str, dict[str, float | None]]:
    deltas: dict[str, dict[str, float | None]] = {}
    for kpi in _PRIMARY_KPIS.get(module, []):
        base_val = baseline_summary.get(kpi)
        scen_val = scenario_summary.get(kpi)
        if isinstance(base_val, (int, float)) and isinstance(scen_val, (int, float)):
            diff = round(float(scen_val) - float(base_val), 4)
            pct = round((diff / float(base_val)) * 100.0, 2) if base_val != 0 else None
            deltas[kpi] = {
                "baseline": round(float(base_val), 4),
                "scenario": round(float(scen_val), 4),
                "delta": diff,
                "delta_pct": pct,
            }
    return deltas


def run_what_if_scenarios(
    module: str,
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    base_constraints: dict[str, Any],
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run Baseline + up to 5 What-If scenarios deterministically and diff KPIs side-by-side."""
    t0 = time.perf_counter()

    if module not in _PRIMARY_KPIS:
        raise ValueError(
            f"Unsupported module '{module}'. Choose from: {sorted(_PRIMARY_KPIS.keys())}."
        )
    if not scenarios:
        raise ValueError("Provide at least 1 scenario to compare against Baseline.")
    if len(scenarios) > MAX_SCENARIOS:
        raise ValueError(
            f"Too many scenarios ({len(scenarios)}). Maximum allowed per call is {MAX_SCENARIOS}."
        )

    parsed_scenarios = [ScenarioSpec.model_validate(s) for s in scenarios]
    base_df = load_dataframe(source_config)

    # 1. Solve Baseline
    baseline_result = _dispatch_solver(module, source_config, mapping, base_constraints, base_df)
    baseline_summary = baseline_result.get("summary", {})

    comparison_table: list[dict[str, Any]] = [
        {
            "scenario": "Baseline",
            "status": baseline_result.get("status"),
            "solve_time_ms": baseline_result.get("solve_time_ms"),
            **{k: baseline_summary.get(k) for k in _PRIMARY_KPIS[module]},
        }
    ]

    scenario_results: list[dict[str, Any]] = []

    # 2. Solve each What-If scenario
    for spec in parsed_scenarios:
        merged_constraints = copy.deepcopy(base_constraints)
        merged_constraints.update(spec.constraint_overrides)
        scen_df = _apply_data_multipliers(base_df, mapping, spec.data_multipliers)

        res = _dispatch_solver(module, source_config, mapping, merged_constraints, scen_df)
        scen_summary = res.get("summary", {})
        deltas = _compute_kpi_deltas(module, baseline_summary, scen_summary)

        scenario_results.append(
            {
                "name": spec.name,
                "description": spec.description,
                "status": res.get("status"),
                "constraint_overrides": spec.constraint_overrides,
                "data_multipliers": spec.data_multipliers,
                "solve_time_ms": res.get("solve_time_ms"),
                "summary": scen_summary,
                "deltas_vs_baseline": deltas,
                "diagnostics": res.get("diagnostics"),
            }
        )

        comparison_table.append(
            {
                "scenario": spec.name,
                "status": res.get("status"),
                "solve_time_ms": res.get("solve_time_ms"),
                **{k: scen_summary.get(k) for k in _PRIMARY_KPIS[module]},
            }
        )

    total_time_ms = (time.perf_counter() - t0) * 1000.0
    audit_trail = build_audit_trail(
        module=f"what_if_{module}",
        solver_used=str(baseline_result.get("solver_used") or "Multi-Solver"),
        status="Completed",
        solve_time_ms=total_time_ms,
        num_variables=0,
        num_constraints=0,
        input_payload={
            "module": module,
            "base_constraints": base_constraints,
            "scenarios": [s.model_dump() for s in parsed_scenarios],
        },
        extra={"runs_executed": 1 + len(parsed_scenarios)},
    )

    return {
        "module": module,
        "total_time_ms": round(total_time_ms, 2),
        "baseline": {
            "status": baseline_result.get("status"),
            "solver_used": baseline_result.get("solver_used"),
            "solve_time_ms": baseline_result.get("solve_time_ms"),
            "summary": baseline_summary,
            "diagnostics": baseline_result.get("diagnostics"),
        },
        "scenarios": scenario_results,
        "comparison_table": comparison_table,
        "audit_trail": audit_trail,
    }
