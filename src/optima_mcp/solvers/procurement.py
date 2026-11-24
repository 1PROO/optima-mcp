"""Deterministic Procurement & Inventory MILP Solver powered by HiGHS (with CBC fallback)."""

from __future__ import annotations

import contextlib
import math
import os
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import pandas as pd
import pulp

from optima_mcp.connectors.base import load_dataframe
from optima_mcp.validation.audit import build_audit_trail, diagnose_procurement_infeasibility
from optima_mcp.validation.schemas import (
    ProcurementConstraints,
    ProcurementItem,
    ProcurementMapping,
    parse_procurement_dataframe,
)

_FD_STDOUT_LOCK = threading.Lock()


@contextlib.contextmanager
def suppress_c_stdout() -> Iterator[None]:
    """Silence C/C++ stdout banners (e.g. highspy) so MCP stdio JSON-RPC stays uncorrupted."""
    with _FD_STDOUT_LOCK:
        saved_fd: int | None = None
        devnull_fd: int | None = None
        try:
            sys.stdout.flush()
            saved_fd = os.dup(1)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(devnull_fd, 1)
        except OSError:
            saved_fd = None
        try:
            yield
        finally:
            if saved_fd is not None:
                try:
                    os.dup2(saved_fd, 1)
                finally:
                    os.close(saved_fd)
            if devnull_fd is not None:
                os.close(devnull_fd)


def add_lp_variable(
    prob: pulp.LpProblem,
    name: str,
    low_bound: float | None = None,
    up_bound: float | None = None,
    cat: str = pulp.LpContinuous,
) -> pulp.LpVariable:
    """Create a PuLP variable compatible with both PuLP 4.x (Rust core) and PuLP 2.x/3.x."""
    if hasattr(prob, "add_variable"):
        return prob.add_variable(name, lowBound=low_bound, upBound=up_bound, cat=cat)
    return pulp.LpVariable(name, lowBound=low_bound, upBound=up_bound, cat=cat)


def count_lp_model_size(prob: pulp.LpProblem) -> tuple[int, int]:
    """Return (num_variables, num_constraints) across PuLP 2.x/3.x and PuLP 4.x."""
    if hasattr(prob, "numVariables") and callable(prob.numVariables):
        num_vars = int(prob.numVariables())
    else:
        num_vars = len(prob.variables())

    if hasattr(prob, "numConstraints") and callable(prob.numConstraints):
        num_cons = int(prob.numConstraints())
    elif callable(prob.constraints):
        num_cons = len(prob.constraints())
    else:
        num_cons = len(prob.constraints)
    return num_vars, num_cons


def solve_lp_silently(prob: pulp.LpProblem, solver: pulp.LpSolver) -> str:
    """Solve a PuLP problem without stdout leaks and return normalized status string."""
    with suppress_c_stdout():
        result = prob.solve(solver)

    # PuLP 4.x returns LpSolveStats with `.status_str`
    if hasattr(result, "status_str"):
        return str(result.status_str)
    # PuLP 2.x/3.x returns an integer status code indexed by pulp.LpStatus
    if hasattr(pulp, "LpStatus") and isinstance(result, int):
        return str(pulp.LpStatus.get(result, "Unknown"))
    return str(result)


def get_milp_solver() -> tuple[pulp.LpSolver, str]:
    """Return HiGHS solver via PuLP (in-memory highspy), falling back to CBC if unavailable."""
    if hasattr(pulp, "HiGHS"):
        try:
            solver = pulp.HiGHS(msg=False)
            if solver.available():
                return solver, "HiGHS"
        except Exception:
            pass

    if hasattr(pulp, "HiGHS_CMD"):
        try:
            solver = pulp.HiGHS_CMD(msg=False)
            if solver.available():
                return solver, "HiGHS"
        except Exception:
            pass

    return pulp.PULP_CBC_CMD(msg=False), "CBC"


def _effective_demand(item: ProcurementItem, planning_horizon_days: float) -> float:
    if item.shelf_life_days is not None and item.shelf_life_days < planning_horizon_days:
        return item.expected_demand * (item.shelf_life_days / planning_horizon_days)
    return item.expected_demand


def _build_procurement_model(
    items: list[ProcurementItem],
    constraints: ProcurementConstraints,
    *,
    objective_override: str | None = None,
    roi_lambda: float | None = None,
    min_spend_floor: float | None = None,
) -> tuple[
    pulp.LpProblem,
    dict[str, pulp.LpVariable],
    dict[str, pulp.LpVariable],
    dict[str, pulp.LpVariable],
    dict[str, tuple[pulp.LpVariable, float]],
    dict[str, float],
    dict[str, float],
]:
    """Build the MILP formulation for procurement optimization."""
    obj_mode = objective_override or constraints.objective
    sense = pulp.LpMinimize if obj_mode == "minimize_cost" else pulp.LpMaximize
    prob = pulp.LpProblem("optima_procurement", sense)

    x_vars: dict[str, pulp.LpVariable] = {}
    y_vars: dict[str, pulp.LpVariable] = {}
    b_vars: dict[str, pulp.LpVariable] = {}
    eff_demands: dict[str, float] = {}
    needs: dict[str, float] = {}

    # Group fixed shipping costs by supplier
    supplier_fees: dict[str, float] = {}
    item_supplier_key: dict[str, str | None] = {}
    for it in items:
        if it.supplier_id:
            sup_key = it.supplier_id
            item_supplier_key[it.item_id] = sup_key
            if it.supplier_fixed_shipping_cost > 0:
                supplier_fees[sup_key] = max(
                    supplier_fees.get(sup_key, 0.0), it.supplier_fixed_shipping_cost
                )
        elif it.supplier_fixed_shipping_cost > 0:
            sup_key = f"__item_{it.item_id}"
            item_supplier_key[it.item_id] = sup_key
            supplier_fees[sup_key] = it.supplier_fixed_shipping_cost
        else:
            item_supplier_key[it.item_id] = None

    z_vars: dict[str, tuple[pulp.LpVariable, float]] = {}
    for sup_key, fee in supplier_fees.items():
        z_vars[sup_key] = (
            add_lp_variable(prob, f"ship_{sup_key}", low_bound=0, up_bound=1, cat=pulp.LpBinary),
            fee,
        )

    for idx, it in enumerate(items):
        eff_d = _effective_demand(it, constraints.planning_horizon_days)
        eff_demands[it.item_id] = eff_d
        need = max(0.0, eff_d - it.current_stock)
        needs[it.item_id] = need

        # Tight upper bound on purchase quantity
        if it.max_order_qty is not None:
            ub = int(it.max_order_qty)
        else:
            # Never need to buy beyond max(ceil(need), moq) since excess cannot be sold
            ub = max(int(math.ceil(need)), int(it.min_order_qty))

        if it.cost > 0:
            budget_cap_qty = int(math.floor(constraints.max_budget / it.cost))
            ub = min(ub, budget_cap_qty)

        if ub < it.min_order_qty:
            # Cannot satisfy MOQ within bounds -> cannot order this item
            ub = 0

        x = add_lp_variable(prob, f"x_{idx}", low_bound=0, up_bound=ub, cat=pulp.LpInteger)
        y = add_lp_variable(prob, f"y_{idx}", low_bound=0, up_bound=need, cat=pulp.LpContinuous)
        b = add_lp_variable(prob, f"b_{idx}", low_bound=0, up_bound=1, cat=pulp.LpBinary)

        x_vars[it.item_id] = x
        y_vars[it.item_id] = y
        b_vars[it.item_id] = b

        # Incremental sold units cannot exceed purchased units (already bounded by need)
        prob += y <= x, f"sell_le_buy_{idx}"

        # Link purchase quantity to binary order indicator & MOQ
        if ub == 0:
            prob += b == 0, f"zero_ub_{idx}"
        else:
            prob += x <= ub * b, f"ub_link_{idx}"
            if it.min_order_qty > 0:
                prob += x >= it.min_order_qty * b, f"moq_{idx}"

        # Link item order to supplier fixed shipping binary
        sup_key = item_supplier_key[it.item_id]
        if sup_key is not None and sup_key in z_vars:
            prob += b <= z_vars[sup_key][0], f"sup_link_{idx}"

    # Spend expression (item costs + supplier fixed shipping fees)
    purchase_cost_expr = pulp.lpSum(it.cost * x_vars[it.item_id] for it in items)
    shipping_cost_expr = pulp.lpSum(fee * z_var for z_var, fee in z_vars.values())
    total_spend_expr = purchase_cost_expr + shipping_cost_expr

    # 1. Budget constraint
    prob += total_spend_expr <= constraints.max_budget, "max_budget"

    if min_spend_floor is not None and min_spend_floor > 0:
        prob += total_spend_expr >= min_spend_floor, "min_spend_floor"

    # 2. Storage constraint (existing stock + new purchases)
    if constraints.max_storage_space is not None:
        current_space = sum(it.current_stock * it.unit_space for it in items)
        add_space_expr = pulp.lpSum(it.unit_space * x_vars[it.item_id] for it in items)
        prob += current_space + add_space_expr <= constraints.max_storage_space, "max_storage_space"

    # 3. Minimum demand coverage constraint
    if constraints.min_demand_coverage > 0:
        if constraints.coverage_mode == "per_item":
            for idx, it in enumerate(items):
                eff_d = eff_demands[it.item_id]
                already = min(it.current_stock, eff_d)
                req_total = constraints.min_demand_coverage * eff_d
                if req_total > already:
                    prob += already + y_vars[it.item_id] >= req_total, f"cov_item_{idx}"
        else:
            total_eff_d = sum(eff_demands.values())
            already_cov = sum(min(it.current_stock, eff_demands[it.item_id]) for it in items)
            target_cov = constraints.min_demand_coverage * total_eff_d
            if target_cov > already_cov:
                prob += (
                    already_cov + pulp.lpSum(y_vars[it.item_id] for it in items) >= target_cov,
                    "min_demand_coverage_agg",
                )

    # 4. Objective function
    revenue_expr = pulp.lpSum(it.selling_price * y_vars[it.item_id] for it in items)
    profit_expr = revenue_expr - total_spend_expr

    if obj_mode == "maximize_profit":
        prob += profit_expr
    elif obj_mode == "minimize_cost":
        prob += total_spend_expr
    elif obj_mode == "maximize_roi":
        lam = 0.0 if roi_lambda is None else roi_lambda
        prob += profit_expr - lam * total_spend_expr
    else:
        raise ValueError(f"Unsupported objective '{obj_mode}'.")

    return prob, x_vars, y_vars, b_vars, z_vars, eff_demands, needs


def solve_procurement(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
    *,
    df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Solve deterministic procurement & inventory replenishment via HiGHS MILP."""
    t0 = time.perf_counter()

    p_map = ProcurementMapping.model_validate(mapping)
    p_cons = ProcurementConstraints.model_validate(constraints)
    df = df_override if df_override is not None else load_dataframe(source_config)

    items, parse_errors = parse_procurement_dataframe(df, p_map, p_cons)
    if parse_errors or not items:
        return {
            "status": "ValidationError",
            "errors": parse_errors or ["No valid items found in dataset."],
            "solver_used": None,
            "solve_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    solver, solver_name = get_milp_solver()
    dinkelbach_iterations = 1

    if p_cons.objective == "maximize_roi":
        # Exact fractional MILP via Dinkelbach's algorithm: maximize Profit / Spend
        positive_costs = [
            it.cost * max(1, it.min_order_qty)
            for it in items
            if it.cost > 0 and _effective_demand(it, p_cons.planning_horizon_days) > it.current_stock
        ]
        min_spend_floor = min(positive_costs) if positive_costs else 1e-4

        lam = 0.0
        best_pack = None
        final_status = "Infeasible"

        for it_num in range(1, 21):
            dinkelbach_iterations = it_num
            prob, x_vars, y_vars, b_vars, z_vars, eff_demands, needs = _build_procurement_model(
                items,
                p_cons,
                objective_override="maximize_roi",
                roi_lambda=lam,
                min_spend_floor=min_spend_floor,
            )
            status_str = solve_lp_silently(prob, solver)
            final_status = status_str

            if status_str != "Optimal":
                # If no positive-spend solution exists, try normal maximize_profit to check feasibility
                prob, x_vars, y_vars, b_vars, z_vars, eff_demands, needs = _build_procurement_model(
                    items, p_cons, objective_override="maximize_profit"
                )
                final_status = solve_lp_silently(prob, solver)
                best_pack = (prob, x_vars, y_vars, b_vars, z_vars, eff_demands, needs)
                break

            spend_val = sum(it.cost * (pulp.value(x_vars[it.item_id]) or 0.0) for it in items) + sum(
                fee * (pulp.value(z_var) or 0.0) for z_var, fee in z_vars.values()
            )
            rev_val = sum(it.selling_price * (pulp.value(y_vars[it.item_id]) or 0.0) for it in items)
            profit_val = rev_val - spend_val
            f_val = profit_val - lam * spend_val
            best_pack = (prob, x_vars, y_vars, b_vars, z_vars, eff_demands, needs)

            if spend_val <= 1e-9 or abs(f_val) <= 1e-6:
                break
            new_lam = profit_val / spend_val
            if abs(new_lam - lam) <= 1e-7:
                break
            lam = new_lam

        assert best_pack is not None
        prob, x_vars, y_vars, b_vars, z_vars, eff_demands, needs = best_pack
        status_str = final_status
    else:
        prob, x_vars, y_vars, b_vars, z_vars, eff_demands, needs = _build_procurement_model(items, p_cons)
        status_str = solve_lp_silently(prob, solver)

    solve_time_ms = (time.perf_counter() - t0) * 1000.0
    num_vars, num_cons = count_lp_model_size(prob)

    input_fingerprint_payload = {
        "module": "procurement",
        "constraints": p_cons.model_dump(),
        "items": [it.model_dump() for it in items],
    }

    if status_str != "Optimal":
        diagnostics = diagnose_procurement_infeasibility(items, p_cons)
        audit_trail = build_audit_trail(
            module="procurement",
            solver_used=solver_name,
            status=status_str,
            solve_time_ms=solve_time_ms,
            num_variables=num_vars,
            num_constraints=num_cons,
            input_payload=input_fingerprint_payload,
            objective_name=p_cons.objective,
            objective_value=None,
        )
        return {
            "status": status_str,
            "solver_used": solver_name,
            "solve_time_ms": round(solve_time_ms, 2),
            "summary": {},
            "purchase_plan": [],
            "skipped_opportunities": [],
            "diagnostics": diagnostics,
            "audit_trail": audit_trail,
        }

    # Extract optimal solution
    purchase_plan: list[dict[str, Any]] = []
    skipped_opportunities: list[dict[str, Any]] = []

    total_item_spend = 0.0
    total_revenue = 0.0
    total_add_space = 0.0
    total_eff_demand = sum(eff_demands.values())
    total_demand_covered = 0.0

    current_space = sum(it.current_stock * it.unit_space for it in items)
    total_shipping_cost = sum(
        fee for z_var, fee in z_vars.values() if (pulp.value(z_var) or 0.0) > 0.5
    )

    for it in items:
        qty = int(round(pulp.value(x_vars[it.item_id]) or 0.0))
        sold_inc = float(pulp.value(y_vars[it.item_id]) or 0.0)
        eff_d = eff_demands[it.item_id]
        already_cov = min(it.current_stock, eff_d)
        total_demand_covered += already_cov + sold_inc

        if qty > 0:
            order_cost = qty * it.cost
            item_rev = sold_inc * it.selling_price
            item_profit = item_rev - order_cost
            margin_pct = ((it.selling_price - it.cost) / it.cost * 100.0) if it.cost > 0 else 0.0
            space_added = qty * it.unit_space

            total_item_spend += order_cost
            total_revenue += item_rev
            total_add_space += space_added

            purchase_plan.append(
                {
                    "item_id": it.item_id,
                    "supplier_id": it.supplier_id,
                    "order_qty": qty,
                    "unit_cost": round(it.cost, 4),
                    "selling_price": round(it.selling_price, 4),
                    "order_cost": round(order_cost, 2),
                    "expected_units_sold": round(sold_inc, 2),
                    "expected_revenue": round(item_rev, 2),
                    "expected_profit": round(item_profit, 2),
                    "margin_pct": round(margin_pct, 2),
                    "post_order_stock": round(it.current_stock + qty, 2),
                    "space_added": round(space_added, 2),
                }
            )
        else:
            need = needs[it.item_id]
            unit_margin = it.selling_price - it.cost
            if need <= 1e-6:
                if it.shelf_life_days is not None and it.shelf_life_days < p_cons.planning_horizon_days:
                    reason = (
                        f"Existing stock ({it.current_stock:g}) already covers shelf-life-capped "
                        f"demand ({eff_d:.1f} over {it.shelf_life_days:g} days)."
                    )
                else:
                    reason = (
                        f"Sufficient current_stock ({it.current_stock:g}) covers expected_demand "
                        f"({it.expected_demand:g})."
                    )
            elif unit_margin <= 0:
                reason = (
                    f"Negative or zero unit margin (cost={it.cost:g} >= selling_price={it.selling_price:g})."
                )
            elif it.max_order_qty is not None and it.max_order_qty == 0:
                reason = "max_order_qty is 0."
            elif it.min_order_qty > 0 and it.min_order_qty * it.cost > p_cons.max_budget:
                reason = (
                    f"Minimum order cost ({it.min_order_qty * it.cost:,.2f}) exceeds max_budget "
                    f"({p_cons.max_budget:,.2f})."
                )
            elif it.min_order_qty > need and (it.selling_price * need - it.cost * it.min_order_qty) <= 0:
                reason = (
                    f"MOQ ({it.min_order_qty}) exceeds unmet demand ({need:.1f}), causing unsold excess "
                    "to make net order profit non-positive."
                )
            else:
                reason = (
                    "Outcompeted by higher profit-per-dollar / profit-per-space SKUs under active "
                    "budget, storage, or supplier-shipping constraints."
                )

            skipped_opportunities.append(
                {
                    "item_id": it.item_id,
                    "supplier_id": it.supplier_id,
                    "unmet_demand": round(need, 2),
                    "unit_cost": round(it.cost, 4),
                    "selling_price": round(it.selling_price, 4),
                    "unit_margin": round(unit_margin, 4),
                    "reason": reason,
                }
            )

    purchase_plan.sort(key=lambda r: (r["expected_profit"], r["order_cost"]), reverse=True)

    total_spend = total_item_spend + total_shipping_cost
    expected_profit = total_revenue - total_spend
    roi_pct = (expected_profit / total_spend * 100.0) if total_spend > 0 else 0.0
    budget_util_pct = (total_spend / p_cons.max_budget * 100.0) if p_cons.max_budget > 0 else 0.0
    total_storage_used = current_space + total_add_space
    storage_util_pct = (
        round(total_storage_used / p_cons.max_storage_space * 100.0, 2)
        if p_cons.max_storage_space and p_cons.max_storage_space > 0
        else None
    )
    demand_cov_pct = (
        (total_demand_covered / total_eff_demand * 100.0) if total_eff_demand > 0 else 100.0
    )

    summary = {
        "objective": p_cons.objective,
        "total_spend": round(total_spend, 2),
        "item_purchase_spend": round(total_item_spend, 2),
        "total_shipping_cost": round(total_shipping_cost, 2),
        "max_budget": round(p_cons.max_budget, 2),
        "budget_utilization_pct": round(budget_util_pct, 2),
        "expected_revenue": round(total_revenue, 2),
        "expected_profit": round(expected_profit, 2),
        "roi_pct": round(roi_pct, 2),
        "storage_used": round(total_storage_used, 2),
        "max_storage_space": round(p_cons.max_storage_space, 2)
        if p_cons.max_storage_space is not None
        else None,
        "storage_utilization_pct": storage_util_pct,
        "demand_coverage_pct": round(demand_cov_pct, 2),
        "items_ordered_count": len(purchase_plan),
        "items_skipped_count": len(skipped_opportunities),
    }

    obj_val = (
        roi_pct
        if p_cons.objective == "maximize_roi"
        else (total_spend if p_cons.objective == "minimize_cost" else expected_profit)
    )
    audit_trail = build_audit_trail(
        module="procurement",
        solver_used=solver_name,
        status=status_str,
        solve_time_ms=solve_time_ms,
        num_variables=num_vars,
        num_constraints=num_cons,
        input_payload=input_fingerprint_payload,
        objective_name=p_cons.objective,
        objective_value=obj_val,
        extra={"dinkelbach_iterations": dinkelbach_iterations}
        if p_cons.objective == "maximize_roi"
        else None,
    )

    return {
        "status": status_str,
        "solver_used": solver_name,
        "solve_time_ms": round(solve_time_ms, 2),
        "summary": summary,
        "purchase_plan": purchase_plan,
        "skipped_opportunities": skipped_opportunities,
        "diagnostics": None,
        "audit_trail": audit_trail,
    }
