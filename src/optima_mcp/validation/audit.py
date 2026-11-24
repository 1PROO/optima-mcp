"""Deterministic Audit Trail builder and Infeasibility Diagnostic Engine."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Any

from optima_mcp.validation.schemas import (
    AllocationConstraints,
    AllocationItem,
    ProcurementConstraints,
    ProcurementItem,
    RoutingConstraints,
    RoutingStop,
    SchedulingConstraints,
    StaffMember,
)


def compute_input_fingerprint(payload: Any) -> str:
    """Compute a deterministic SHA-256 fingerprint of normalized solver inputs."""
    canonical = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_audit_trail(
    *,
    module: str,
    solver_used: str,
    status: str,
    solve_time_ms: float,
    num_variables: int,
    num_constraints: int,
    input_payload: Any,
    objective_name: str | None = None,
    objective_value: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a structured, reproducible audit record for every solver execution."""
    record: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "module": module,
        "solver_used": solver_used,
        "status": status,
        "solve_time_ms": round(solve_time_ms, 2),
        "model_size": {
            "variables": num_variables,
            "constraints": num_constraints,
        },
        "objective_name": objective_name,
        "objective_value": round(objective_value, 4) if objective_value is not None else None,
        "input_fingerprint": compute_input_fingerprint(input_payload),
        "deterministic": True,
    }
    if extra:
        record.update(extra)
    return record


# ---------------------------------------------------------------------------
# Infeasibility Diagnostic Engine
# ---------------------------------------------------------------------------


def _effective_demand(item: ProcurementItem, planning_horizon_days: float) -> float:
    if item.shelf_life_days is not None and item.shelf_life_days < planning_horizon_days:
        return item.expected_demand * (item.shelf_life_days / planning_horizon_days)
    return item.expected_demand


def diagnose_procurement_infeasibility(
    items: list[ProcurementItem],
    constraints: ProcurementConstraints,
) -> dict[str, Any]:
    """Pinpoint exact conflicting constraints when procurement MILP is Infeasible."""
    conflicts: list[dict[str, Any]] = []
    recommendations: list[str] = []

    # 1. Existing stock vs storage cap
    current_storage = sum(it.current_stock * it.unit_space for it in items)
    if constraints.max_storage_space is not None and current_storage > constraints.max_storage_space + 1e-6:
        deficit = current_storage - constraints.max_storage_space
        conflicts.append(
            {
                "constraint": "max_storage_space",
                "reason": (
                    f"Existing current_stock uses {current_storage:,.2f} storage units, already exceeding "
                    f"max_storage_space ({constraints.max_storage_space:,.2f}) by {deficit:,.2f}."
                ),
                "minimum_required_value": round(current_storage, 2),
            }
        )
        recommendations.append(
            f"Increase max_storage_space to at least {current_storage:,.2f} (plus room for new orders)."
        )

    # 2. Demand coverage vs max_order_qty / budget / storage
    if constraints.min_demand_coverage > 0:
        eff_demands = [_effective_demand(it, constraints.planning_horizon_days) for it in items]
        total_eff_demand = sum(eff_demands)
        target_covered = constraints.min_demand_coverage * total_eff_demand
        already_covered = sum(min(it.current_stock, d) for it, d in zip(items, eff_demands))
        additional_needed = max(0.0, target_covered - already_covered)

        # Max achievable additional units respecting max_order_qty
        max_add_units = 0.0
        for it, d in zip(items, eff_demands):
            need = max(0.0, d - it.current_stock)
            cap = float(it.max_order_qty) if it.max_order_qty is not None else need
            max_add_units += min(need, cap)

        if additional_needed > max_add_units + 1e-6:
            max_possible_cov = (already_covered + max_add_units) / total_eff_demand if total_eff_demand > 0 else 1.0
            conflicts.append(
                {
                    "constraint": "min_demand_coverage vs max_order_qty",
                    "reason": (
                        f"min_demand_coverage={constraints.min_demand_coverage:.2%} requires "
                        f"{additional_needed:,.1f} additional units, but max_order_qty caps total "
                        f"purchasable demand-covering units at {max_add_units:,.1f} "
                        f"(max achievable coverage: {max_possible_cov:.2%})."
                    ),
                    "max_achievable_coverage": round(max_possible_cov, 4),
                }
            )
            recommendations.append(
                f"Lower min_demand_coverage to <= {max_possible_cov:.2%} or raise max_order_qty."
            )

        # Greedy lower bound on minimum spend & storage required to hit additional_needed
        if additional_needed > 0:
            # Lower bound on spend: sort by cheapest cost per unit
            by_cost = sorted(
                [(it.cost, max(0.0, d - it.current_stock), it.min_order_qty) for it, d in zip(items, eff_demands)],
                key=lambda x: x[0],
            )
            rem = additional_needed
            min_spend_lb = 0.0
            for unit_cost, need, _ in by_cost:
                if rem <= 0:
                    break
                take = min(rem, need)
                min_spend_lb += take * unit_cost
                rem -= take

            if min_spend_lb > constraints.max_budget + 1e-6:
                conflicts.append(
                    {
                        "constraint": "max_budget vs min_demand_coverage",
                        "reason": (
                            f"Covering {constraints.min_demand_coverage:.2%} of demand requires at least "
                            f"{min_spend_lb:,.2f} in spend (even ignoring MOQ/shipping), exceeding "
                            f"max_budget ({constraints.max_budget:,.2f})."
                        ),
                        "minimum_budget_lower_bound": round(min_spend_lb, 2),
                    }
                )
                recommendations.append(
                    f"Increase max_budget above {min_spend_lb:,.2f} or reduce min_demand_coverage."
                )

            # Lower bound on storage required
            if constraints.max_storage_space is not None:
                by_space = sorted(
                    [(it.unit_space, max(0.0, d - it.current_stock)) for it, d in zip(items, eff_demands)],
                    key=lambda x: x[0],
                )
                rem = additional_needed
                min_add_space = 0.0
                for us, need in by_space:
                    if rem <= 0:
                        break
                    take = min(rem, need)
                    min_add_space += take * us
                    rem -= take
                total_space_lb = current_storage + min_add_space
                if total_space_lb > constraints.max_storage_space + 1e-6:
                    conflicts.append(
                        {
                            "constraint": "max_storage_space vs min_demand_coverage",
                            "reason": (
                                f"Covering {constraints.min_demand_coverage:.2%} of demand requires at least "
                                f"{total_space_lb:,.2f} total storage units, exceeding max_storage_space "
                                f"({constraints.max_storage_space:,.2f})."
                            ),
                            "minimum_storage_lower_bound": round(total_space_lb, 2),
                        }
                    )
                    recommendations.append(
                        f"Increase max_storage_space above {total_space_lb:,.2f} or lower min_demand_coverage."
                    )

        # Check per-item coverage conflicts specifically
        if constraints.coverage_mode == "per_item":
            per_item_cost = 0.0
            for it, d in zip(items, eff_demands):
                req_units = max(0.0, constraints.min_demand_coverage * d - it.current_stock)
                if req_units > 0:
                    order_min = max(req_units, float(it.min_order_qty))
                    per_item_cost += order_min * it.cost
                    if it.max_order_qty is not None and req_units > it.max_order_qty:
                        conflicts.append(
                            {
                                "constraint": f"per_item_coverage({it.item_id})",
                                "reason": (
                                    f"Item '{it.item_id}' needs {req_units:.1f} units for coverage "
                                    f"but max_order_qty is {it.max_order_qty}."
                                ),
                            }
                        )
            if per_item_cost > constraints.max_budget + 1e-6:
                conflicts.append(
                    {
                        "constraint": "max_budget vs per_item min_demand_coverage",
                        "reason": (
                            f"Per-item demand coverage requires at least {per_item_cost:,.2f} in spend, "
                            f"exceeding max_budget ({constraints.max_budget:,.2f})."
                        ),
                        "minimum_budget_lower_bound": round(per_item_cost, 2),
                    }
                )

    if not conflicts:
        conflicts.append(
            {
                "constraint": "coupled_constraints",
                "reason": (
                    "Combined interaction of max_budget, max_storage_space, MOQ/shipping fees, "
                    "and min_demand_coverage leaves no integer feasible solution."
                ),
            }
        )
        recommendations.append(
            "Use run_what_if_scenarios to relax max_budget, max_storage_space, or min_demand_coverage."
        )

    return {
        "feasible": False,
        "conflicting_constraints": conflicts,
        "recommendations": recommendations,
    }


def diagnose_routing_infeasibility(
    stops: list[RoutingStop],
    constraints: RoutingConstraints,
    distance_matrix_km: list[list[float]],
    depot_idx: int = 0,
) -> dict[str, Any]:
    """Pinpoint exact causes of VRP infeasibility (capacity or max_route_distance)."""
    conflicts: list[dict[str, Any]] = []
    recommendations: list[str] = []

    caps = constraints.vehicle_capacities or []
    total_cap = sum(caps)
    max_cap = max(caps) if caps else 0.0
    non_depot_stops = [s for idx, s in enumerate(stops) if idx != depot_idx]
    total_demand = sum(s.demand for s in non_depot_stops)

    if total_demand > total_cap + 1e-6:
        conflicts.append(
            {
                "constraint": "vehicle_capacities",
                "reason": (
                    f"Total delivery demand ({total_demand:,.2f}) exceeds total fleet capacity "
                    f"({total_cap:,.2f}) by {total_demand - total_cap:,.2f}."
                ),
                "minimum_required_capacity": round(total_demand, 2),
            }
        )
        recommendations.append(
            f"Add vehicles or increase total vehicle capacity to at least {total_demand:,.2f}, "
            "or set allow_drop_stops=True."
        )

    oversized = [s.stop_id for s in non_depot_stops if s.demand > max_cap + 1e-6]
    if oversized:
        conflicts.append(
            {
                "constraint": "single_vehicle_capacity",
                "reason": (
                    f"Stop(s) {oversized[:5]} have individual demand exceeding the largest vehicle "
                    f"capacity ({max_cap:,.2f})."
                ),
            }
        )
        recommendations.append("Split oversized stop deliveries or assign a larger-capacity vehicle.")

    if constraints.max_route_distance is not None and distance_matrix_km:
        unreachable: list[tuple[str, float]] = []
        for idx, stop in enumerate(stops):
            if idx == depot_idx:
                continue
            round_trip = distance_matrix_km[depot_idx][idx] + distance_matrix_km[idx][depot_idx]
            if round_trip > constraints.max_route_distance + 1e-6:
                unreachable.append((stop.stop_id, round(round_trip, 3)))
        if unreachable:
            max_rt = max(rt for _, rt in unreachable)
            conflicts.append(
                {
                    "constraint": "max_route_distance",
                    "reason": (
                        f"{len(unreachable)} stop(s) exceed max_route_distance ({constraints.max_route_distance:,.2f}) "
                        f"even on a direct round-trip from depot (max round-trip: {max_rt:,.2f})."
                    ),
                    "unreachable_stops_sample": [u[0] for u in unreachable[:5]],
                    "minimum_required_distance": max_rt,
                }
            )
            recommendations.append(
                f"Increase max_route_distance above {max_rt:,.2f} or enable allow_drop_stops=True."
            )

    if not conflicts:
        conflicts.append(
            {
                "constraint": "coupled_capacity_and_distance",
                "reason": (
                    "Fleet capacity packing combined with max_route_distance / vehicle count cannot serve all stops."
                ),
            }
        )
        recommendations.append(
            "Increase num_vehicles, max_route_distance, or run_what_if_scenarios to test relaxed fleet limits."
        )

    return {
        "feasible": False,
        "conflicting_constraints": conflicts,
        "recommendations": recommendations,
    }


def diagnose_scheduling_infeasibility(
    staff: list[StaffMember],
    constraints: SchedulingConstraints,
    slot_requirements: dict[tuple[str, str], int],
    slot_skill_requirements: dict[tuple[str, str], dict[str, int]],
) -> dict[str, Any]:
    """Pinpoint exact causes of CP-SAT scheduling infeasibility."""
    conflicts: list[dict[str, Any]] = []
    recommendations: list[str] = []

    total_required = sum(slot_requirements.values())
    total_capacity = sum(st.max_shifts for st in staff)

    if total_required > total_capacity:
        conflicts.append(
            {
                "constraint": "max_shifts (total staff capacity)",
                "reason": (
                    f"Total required shift slots ({total_required}) exceeds sum of staff max_shifts "
                    f"({total_capacity}) by {total_required - total_capacity} shift(s)."
                ),
                "shortfall_shifts": total_required - total_capacity,
            }
        )
        recommendations.append(
            f"Add staff or increase max_shifts across the team by at least {total_required - total_capacity} shifts."
        )

    # Check single-day headcount vs available staff on that day
    days = constraints.day_names()
    for day in days:
        avail_staff = [st for st in staff if day not in st.unavailable_days and st.max_shifts > 0]
        day_req = sum(slot_requirements.get((day, sh), 0) for sh in constraints.shifts)
        day_max_assignable = len(avail_staff) * constraints.max_shifts_per_day
        if day_req > day_max_assignable:
            conflicts.append(
                {
                    "constraint": f"daily_capacity({day})",
                    "reason": (
                        f"Day '{day}' requires {day_req} shift slots, but only {len(avail_staff)} "
                        f"available staff * max_shifts_per_day ({constraints.max_shifts_per_day}) = "
                        f"{day_max_assignable}."
                    ),
                }
            )
            recommendations.append(f"Reduce headcount on '{day}' or increase available staff.")

    # Check skill requirements
    skill_totals: dict[str, int] = {}
    for (day, sh), sk_req in slot_skill_requirements.items():
        for skill, count in sk_req.items():
            skill_totals[skill] = skill_totals.get(skill, 0) + count
            qualified_today = [
                st
                for st in staff
                if skill in st.skills and day not in st.unavailable_days and st.max_shifts > 0
            ]
            if count > len(qualified_today):
                conflicts.append(
                    {
                        "constraint": f"required_skills({day}, {sh}, {skill})",
                        "reason": (
                            f"Shift ({day}, {sh}) requires {count} staff with skill '{skill}', "
                            f"but only {len(qualified_today)} qualified staff are available."
                        ),
                    }
                )

    for skill, total_needed in skill_totals.items():
        skill_cap = sum(st.max_shifts for st in staff if skill in st.skills)
        if total_needed > skill_cap:
            conflicts.append(
                {
                    "constraint": f"total_skill_capacity({skill})",
                    "reason": (
                        f"Skill '{skill}' requires {total_needed} total shift slots across the horizon, "
                        f"but qualified staff only have {skill_cap} combined max_shifts."
                    ),
                }
            )
            recommendations.append(
                f"Cross-train or hire additional staff with skill '{skill}' (shortfall: {total_needed - skill_cap} shifts)."
            )

    if not conflicts:
        conflicts.append(
            {
                "constraint": "rest_rule_or_coupled_constraints",
                "reason": (
                    "No-back-to-back rest rules combined with max_shifts_per_day, unavailability, "
                    "and skill requirements leave no valid assignment."
                ),
            }
        )
        recommendations.append(
            "Test relaxing no_back_to_back, lowering off-peak required_headcount, or raising max_shifts via run_what_if_scenarios."
        )

    return {
        "feasible": False,
        "conflicting_constraints": conflicts,
        "recommendations": recommendations,
    }


def diagnose_allocation_infeasibility(
    items: list[AllocationItem],
    constraints: AllocationConstraints,
) -> dict[str, Any]:
    """Pinpoint exact causes of allocation LP/MILP infeasibility."""
    conflicts: list[dict[str, Any]] = []
    recommendations: list[str] = []

    min_spend = sum(it.min_allocation * it.unit_cost for it in items)
    if min_spend > constraints.total_budget + 1e-6:
        conflicts.append(
            {
                "constraint": "min_allocation vs total_budget",
                "reason": (
                    f"Mandatory minimum allocations require {min_spend:,.2f}, exceeding "
                    f"total_budget ({constraints.total_budget:,.2f}) by {min_spend - constraints.total_budget:,.2f}."
                ),
                "minimum_required_budget": round(min_spend, 2),
            }
        )
        recommendations.append(
            f"Increase total_budget to at least {min_spend:,.2f} or lower item min_allocation bounds."
        )

    if constraints.category_quotas:
        cat_min_sum = 0.0
        for cat, quota in constraints.category_quotas.items():
            cat_items = [it for it in items if it.category == cat]
            req_min = max(
                float(quota.get("min", 0.0)),
                float(quota.get("min_pct", 0.0)) * constraints.total_budget,
            )
            cat_min_sum += req_min
            if not cat_items and req_min > 0:
                conflicts.append(
                    {
                        "constraint": f"category_quota({cat})",
                        "reason": f"Category '{cat}' has minimum quota {req_min:,.2f} but 0 items belong to it.",
                    }
                )
            else:
                if all(it.max_allocation is not None for it in cat_items):
                    cat_max_cap = sum((it.max_allocation or 0.0) * it.unit_cost for it in cat_items)
                    if req_min > cat_max_cap + 1e-6:
                        conflicts.append(
                            {
                                "constraint": f"category_quota_min({cat}) vs max_allocation",
                                "reason": (
                                    f"Category '{cat}' requires at least {req_min:,.2f}, but item max_allocation "
                                    f"caps allow at most {cat_max_cap:,.2f}."
                                ),
                            }
                        )
                req_max = min(
                    float(quota.get("max", float("inf"))),
                    float(quota.get("max_pct", 1.0)) * constraints.total_budget
                    if "max_pct" in quota
                    else float("inf"),
                )
                cat_item_min = sum(it.min_allocation * it.unit_cost for it in cat_items)
                if cat_item_min > req_max + 1e-6:
                    conflicts.append(
                        {
                            "constraint": f"category_quota_max({cat}) vs item min_allocation",
                            "reason": (
                                f"Category '{cat}' max quota is {req_max:,.2f}, but item min_allocation "
                                f"bounds already total {cat_item_min:,.2f}."
                            ),
                        }
                    )

        if cat_min_sum > constraints.total_budget + 1e-6:
            conflicts.append(
                {
                    "constraint": "category_quotas_sum vs total_budget",
                    "reason": (
                        f"Sum of category minimum quotas ({cat_min_sum:,.2f}) exceeds "
                        f"total_budget ({constraints.total_budget:,.2f})."
                    ),
                }
            )
            recommendations.append(
                f"Reduce category minimum quotas or increase total_budget to at least {cat_min_sum:,.2f}."
            )

    if not conflicts:
        conflicts.append(
            {
                "constraint": "coupled_bounds_or_return_target",
                "reason": (
                    "Combination of total_budget, item bounds, category quotas, integer_vars, "
                    "or min_total_return has no feasible solution."
                ),
            }
        )
        recommendations.append("Use run_what_if_scenarios to relax total_budget or category quotas.")

    return {
        "feasible": False,
        "conflicting_constraints": conflicts,
        "recommendations": recommendations,
    }
