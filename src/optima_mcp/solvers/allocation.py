"""Deterministic Resource, Capital, Shelf-Space & Production Mix Allocator (HiGHS LP/MILP)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import pulp

from optima_mcp.connectors.base import load_dataframe
from optima_mcp.solvers.procurement import (
    add_lp_variable,
    count_lp_model_size,
    get_milp_solver,
    solve_lp_silently,
)
from optima_mcp.validation.audit import build_audit_trail, diagnose_allocation_infeasibility
from optima_mcp.validation.schemas import (
    AllocationConstraints,
    AllocationMapping,
    parse_allocation_dataframe,
)


def solve_allocation(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
    *,
    df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Solve general LP/MILP capital, shelf-space, blending, or production mix allocation."""
    t0 = time.perf_counter()

    a_map = AllocationMapping.model_validate(mapping)
    a_cons = AllocationConstraints.model_validate(constraints)
    df = df_override if df_override is not None else load_dataframe(source_config)

    items, parse_errors = parse_allocation_dataframe(df, a_map)
    if parse_errors or not items:
        return {
            "status": "ValidationError",
            "errors": parse_errors or ["No valid allocation items found."],
            "solver_used": None,
            "solve_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    sense = pulp.LpMinimize if a_cons.objective == "minimize_cost" else pulp.LpMaximize
    prob = pulp.LpProblem("optima_allocation", sense)
    var_cat = pulp.LpInteger if a_cons.integer_vars else pulp.LpContinuous

    x_vars: dict[str, pulp.LpVariable] = {}
    for idx, it in enumerate(items):
        ub = it.max_allocation if it.max_allocation is not None else (a_cons.total_budget / it.unit_cost)
        x_vars[it.item_id] = add_lp_variable(
            prob,
            f"alloc_{idx}",
            low_bound=it.min_allocation,
            up_bound=ub,
            cat=var_cat,
        )

    total_spend_expr = pulp.lpSum(it.unit_cost * x_vars[it.item_id] for it in items)
    total_return_expr = pulp.lpSum(it.expected_return * x_vars[it.item_id] for it in items)

    # 1. Total budget / capacity constraint
    prob += total_spend_expr <= a_cons.total_budget, "total_budget"

    # 2. Optional minimum return target (commonly used with minimize_cost)
    if a_cons.min_total_return is not None:
        prob += total_return_expr >= a_cons.min_total_return, "min_total_return"

    # 3. Optional category quotas (supports absolute 'min'/'max' and 'min_pct'/'max_pct' of total_budget)
    if a_cons.category_quotas:
        for cat_name, quota in a_cons.category_quotas.items():
            cat_items = [it for it in items if it.category == cat_name]
            cat_spend_expr = pulp.lpSum(it.unit_cost * x_vars[it.item_id] for it in cat_items)

            if "min" in quota:
                prob += cat_spend_expr >= float(quota["min"]), f"cat_min_{cat_name}"
            if "min_pct" in quota:
                prob += (
                    cat_spend_expr >= float(quota["min_pct"]) * a_cons.total_budget,
                    f"cat_min_pct_{cat_name}",
                )
            if "max" in quota:
                prob += cat_spend_expr <= float(quota["max"]), f"cat_max_{cat_name}"
            if "max_pct" in quota:
                prob += (
                    cat_spend_expr <= float(quota["max_pct"]) * a_cons.total_budget,
                    f"cat_max_pct_{cat_name}",
                )

    # 4. Objective
    if a_cons.objective == "minimize_cost":
        prob += total_spend_expr
    else:
        prob += total_return_expr

    solver, solver_name = get_milp_solver()
    status_str = solve_lp_silently(prob, solver)
    solve_time_ms = (time.perf_counter() - t0) * 1000.0
    num_vars, num_cons = count_lp_model_size(prob)

    input_payload = {
        "module": "allocation",
        "constraints": a_cons.model_dump(),
        "items": [it.model_dump() for it in items],
    }

    if status_str != "Optimal":
        diagnostics = diagnose_allocation_infeasibility(items, a_cons)
        audit_trail = build_audit_trail(
            module="allocation",
            solver_used=solver_name,
            status=status_str,
            solve_time_ms=solve_time_ms,
            num_variables=num_vars,
            num_constraints=num_cons,
            input_payload=input_payload,
            objective_name=a_cons.objective,
        )
        return {
            "status": status_str,
            "solver_used": solver_name,
            "solve_time_ms": round(solve_time_ms, 2),
            "summary": {},
            "allocations": [],
            "skipped_opportunities": [],
            "diagnostics": diagnostics,
            "audit_trail": audit_trail,
        }

    allocations: list[dict[str, Any]] = []
    skipped_opportunities: list[dict[str, Any]] = []
    total_allocated = 0.0
    total_return = 0.0

    for it in items:
        raw_val = float(pulp.value(x_vars[it.item_id]) or 0.0)
        val = float(int(round(raw_val))) if a_cons.integer_vars else raw_val
        spend = val * it.unit_cost
        ret = val * it.expected_return

        if val > 1e-7:
            total_allocated += spend
            total_return += ret
            allocations.append(
                {
                    "item_id": it.item_id,
                    "category": it.category,
                    "allocated_units": round(val, 4),
                    "unit_cost": round(it.unit_cost, 4),
                    "allocated_budget": round(spend, 2),
                    "budget_share_pct": round(spend / a_cons.total_budget * 100.0, 2),
                    "expected_return_per_unit": round(it.expected_return, 4),
                    "expected_return": round(ret, 2),
                }
            )
        else:
            if it.expected_return <= 0:
                reason = f"Non-positive expected_return ({it.expected_return:g})."
            else:
                reason = (
                    f"Lower return-per-cost ratio ({it.expected_return / it.unit_cost:.4f}) "
                    "compared to selected alternatives under total_budget / category quotas."
                )
            skipped_opportunities.append(
                {
                    "item_id": it.item_id,
                    "category": it.category,
                    "expected_return_per_unit": round(it.expected_return, 4),
                    "unit_cost": round(it.unit_cost, 4),
                    "return_per_cost": round(it.expected_return / it.unit_cost, 4),
                    "reason": reason,
                }
            )

    allocations.sort(key=lambda r: (r["expected_return"], r["allocated_budget"]), reverse=True)

    budget_util_pct = (total_allocated / a_cons.total_budget * 100.0) if a_cons.total_budget > 0 else 0.0
    weighted_return_pct = (total_return / total_allocated * 100.0) if total_allocated > 0 else 0.0

    summary = {
        "objective": a_cons.objective,
        "total_budget": round(a_cons.total_budget, 2),
        "total_allocated": round(total_allocated, 2),
        "budget_utilization_pct": round(budget_util_pct, 2),
        "total_expected_return": round(total_return, 2),
        "weighted_return_pct": round(weighted_return_pct, 2),
        "items_funded_count": len(allocations),
        "items_skipped_count": len(skipped_opportunities),
    }

    audit_trail = build_audit_trail(
        module="allocation",
        solver_used=solver_name,
        status=status_str,
        solve_time_ms=solve_time_ms,
        num_variables=num_vars,
        num_constraints=num_cons,
        input_payload=input_payload,
        objective_name=a_cons.objective,
        objective_value=total_allocated if a_cons.objective == "minimize_cost" else total_return,
    )

    return {
        "status": status_str,
        "solver_used": solver_name,
        "solve_time_ms": round(solve_time_ms, 2),
        "summary": summary,
        "allocations": allocations,
        "skipped_opportunities": skipped_opportunities,
        "diagnostics": None,
        "audit_trail": audit_trail,
    }
