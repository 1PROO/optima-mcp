"""Deterministic Workforce & Shift Scheduling Solver powered by Google OR-Tools CP-SAT."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
from ortools.sat.python import cp_model

from optima_mcp.connectors.base import load_dataframe
from optima_mcp.validation.audit import build_audit_trail, diagnose_scheduling_infeasibility
from optima_mcp.validation.schemas import (
    SchedulingConstraints,
    SchedulingMapping,
    parse_scheduling_dataframe,
)


def _resolve_headcount(
    constraints: SchedulingConstraints, days: list[str]
) -> dict[tuple[str, str], int]:
    reqs: dict[tuple[str, str], int] = {}
    rh = constraints.required_headcount
    for d in days:
        for sh in constraints.shifts:
            if isinstance(rh, int):
                reqs[(d, sh)] = rh
            elif d in rh and isinstance(rh[d], dict):
                reqs[(d, sh)] = int(rh[d].get(sh, 0))  # type: ignore[union-attr]
            else:
                reqs[(d, sh)] = int(rh.get(sh, 0))  # type: ignore[arg-type]
    return reqs


def _resolve_skill_requirements(
    constraints: SchedulingConstraints, days: list[str]
) -> dict[tuple[str, str], dict[str, int]]:
    out: dict[tuple[str, str], dict[str, int]] = {}
    rs = constraints.required_skills
    if not rs:
        return out

    # Detect whether rs is {shift_name: {skill: count}} or {skill: count_per_shift}
    first_val = next(iter(rs.values()))
    if isinstance(first_val, dict):
        for d in days:
            for sh in constraints.shifts:
                if sh in rs and isinstance(rs[sh], dict):
                    out[(d, sh)] = {str(k): int(v) for k, v in rs[sh].items() if int(v) > 0}  # type: ignore[union-attr]
    else:
        uniform = {str(k): int(v) for k, v in rs.items() if int(v) > 0}  # type: ignore[arg-type]
        for d in days:
            for sh in constraints.shifts:
                if uniform:
                    out[(d, sh)] = dict(uniform)
    return out


def solve_scheduling(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
    *,
    df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Solve fair shift & workforce scheduling using Google OR-Tools CP-SAT."""
    t0 = time.perf_counter()

    s_map = SchedulingMapping.model_validate(mapping)
    s_cons = SchedulingConstraints.model_validate(constraints)
    df = df_override if df_override is not None else load_dataframe(source_config)

    staff, parse_errors = parse_scheduling_dataframe(df, s_map)
    if parse_errors or not staff:
        return {
            "status": "ValidationError",
            "errors": parse_errors or ["No valid staff records found."],
            "solver_used": None,
            "solve_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    days = s_cons.day_names()
    shifts = s_cons.shifts
    slot_reqs = _resolve_headcount(s_cons, days)
    slot_skill_reqs = _resolve_skill_requirements(s_cons, days)

    model = cp_model.CpModel()
    num_staff = len(staff)
    num_days = len(days)
    num_shifts = len(shifts)

    # Decision variable: work[(e, d, s)] == 1 iff employee e works shift s on day d
    work: dict[tuple[int, int, int], cp_model.IntVar] = {}
    for e in range(num_staff):
        for d in range(num_days):
            for s in range(num_shifts):
                work[(e, d, s)] = model.NewBoolVar(f"w_{e}_{d}_{s}")

    num_constraints = 0

    # 1. Exact headcount per (day, shift)
    for d_idx, d_name in enumerate(days):
        for s_idx, s_name in enumerate(shifts):
            req = slot_reqs[(d_name, s_name)]
            model.Add(sum(work[(e, d_idx, s_idx)] for e in range(num_staff)) == req)
            num_constraints += 1

    # 2. Required skills per (day, shift)
    for d_idx, d_name in enumerate(days):
        for s_idx, s_name in enumerate(shifts):
            sk_req = slot_skill_reqs.get((d_name, s_name), {})
            for skill, min_cnt in sk_req.items():
                qualified = [e for e, st in enumerate(staff) if skill in st.skills]
                model.Add(sum(work[(e, d_idx, s_idx)] for e in qualified) >= min_cnt)
                num_constraints += 1

    # 3. Employee max shifts, daily shift cap, unavailability, and no back-to-back rest rule
    total_shifts_vars: list[cp_model.IntVar] = []
    for e, st in enumerate(staff):
        # Max shifts over horizon
        e_total = model.NewIntVar(0, st.max_shifts, f"total_shifts_{e}")
        model.Add(
            e_total
            == sum(work[(e, d, s)] for d in range(num_days) for s in range(num_shifts))
        )
        total_shifts_vars.append(e_total)
        num_constraints += 1

        # Max shifts per day & unavailability
        for d_idx, d_name in enumerate(days):
            if d_name in st.unavailable_days:
                for s_idx in range(num_shifts):
                    model.Add(work[(e, d_idx, s_idx)] == 0)
                    num_constraints += 1
            else:
                model.Add(
                    sum(work[(e, d_idx, s_idx)] for s_idx in range(num_shifts))
                    <= s_cons.max_shifts_per_day
                )
                num_constraints += 1

        # No back-to-back shifts across the linearized timeline (including Night -> next Morning)
        if s_cons.no_back_to_back:
            timeline = [
                work[(e, d_idx, s_idx)]
                for d_idx in range(num_days)
                for s_idx in range(num_shifts)
            ]
            for t in range(len(timeline) - 1):
                model.Add(timeline[t] + timeline[t + 1] <= 1)
                num_constraints += 1

    # 4. Objective
    cost_scale = 100
    total_cost_expr = sum(
        int(round(st.shift_cost * cost_scale)) * total_shifts_vars[e]
        for e, st in enumerate(staff)
    )

    if s_cons.objective == "minimize_cost":
        model.Minimize(total_cost_expr)
    else:
        # Balance workload: minimize (max_assigned - min_assigned) among eligible staff
        eligible = [e for e, st in enumerate(staff) if st.max_shifts > 0]
        if eligible:
            horizon_slots = num_days * num_shifts
            max_assigned = model.NewIntVar(0, horizon_slots, "max_assigned")
            min_assigned = model.NewIntVar(0, horizon_slots, "min_assigned")
            for e in eligible:
                model.Add(max_assigned >= total_shifts_vars[e])
                model.Add(min_assigned <= total_shifts_vars[e])
            spread = max_assigned - min_assigned
            # Primary: minimize spread; tiny secondary tie-breaker: labor cost
            model.Minimize(spread * 100_000 + total_cost_expr)
        else:
            model.Minimize(0)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(s_cons.time_limit_seconds)
    solver.parameters.num_search_workers = 1  # Strictly deterministic execution
    cp_status = solver.Solve(model)
    solve_time_ms = (time.perf_counter() - t0) * 1000.0

    status_map = {
        cp_model.OPTIMAL: "Optimal",
        cp_model.FEASIBLE: "Feasible",
        cp_model.INFEASIBLE: "Infeasible",
        cp_model.MODEL_INVALID: "ModelInvalid",
        cp_model.UNKNOWN: "Unknown",
    }
    status_str = status_map.get(cp_status, "Unknown")

    input_payload = {
        "module": "scheduling",
        "constraints": s_cons.model_dump(),
        "staff": [st.model_dump() for st in staff],
    }

    if cp_status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        diagnostics = diagnose_scheduling_infeasibility(staff, s_cons, slot_reqs, slot_skill_reqs)
        audit_trail = build_audit_trail(
            module="scheduling",
            solver_used="OR-Tools CP-SAT",
            status=status_str,
            solve_time_ms=solve_time_ms,
            num_variables=len(work) + len(total_shifts_vars),
            num_constraints=num_constraints,
            input_payload=input_payload,
            objective_name=s_cons.objective,
        )
        return {
            "status": status_str,
            "solver_used": "OR-Tools CP-SAT",
            "solve_time_ms": round(solve_time_ms, 2),
            "summary": {},
            "schedule": {},
            "staff_workload": [],
            "diagnostics": diagnostics,
            "audit_trail": audit_trail,
        }

    # Build structured schedule & staff workload
    schedule: dict[str, dict[str, list[str]]] = {}
    staff_slots: dict[int, list[str]] = {e: [] for e in range(num_staff)}
    total_filled = 0

    for d_idx, d_name in enumerate(days):
        schedule[d_name] = {}
        for s_idx, s_name in enumerate(shifts):
            assigned_ids: list[str] = []
            for e, st in enumerate(staff):
                if solver.Value(work[(e, d_idx, s_idx)]) == 1:
                    assigned_ids.append(st.staff_id)
                    staff_slots[e].append(f"{d_name}:{s_name}")
                    total_filled += 1
            schedule[d_name][s_name] = assigned_ids

    staff_workload: list[dict[str, Any]] = []
    total_labor_cost = 0.0
    assigned_counts: list[int] = []

    for e, st in enumerate(staff):
        cnt = len(staff_slots[e])
        assigned_counts.append(cnt)
        cost = cnt * st.shift_cost
        total_labor_cost += cost
        util_pct = (cnt / st.max_shifts * 100.0) if st.max_shifts > 0 else 0.0
        staff_workload.append(
            {
                "staff_id": st.staff_id,
                "name": st.name,
                "skills": st.skills,
                "shifts_assigned": cnt,
                "max_shifts": st.max_shifts,
                "utilization_pct": round(util_pct, 2),
                "labor_cost": round(cost, 2),
                "assigned_slots": staff_slots[e],
            }
        )

    total_required = sum(slot_reqs.values())
    staff_utilized = sum(1 for c in assigned_counts if c > 0)
    min_sh = min(assigned_counts) if assigned_counts else 0
    max_sh = max(assigned_counts) if assigned_counts else 0

    summary = {
        "objective": s_cons.objective,
        "total_required_slots": total_required,
        "total_shifts_filled": total_filled,
        "staff_count": num_staff,
        "staff_utilized": staff_utilized,
        "min_shifts_per_staff": min_sh,
        "max_shifts_per_staff": max_sh,
        "workload_spread": max_sh - min_sh,
        "total_labor_cost": round(total_labor_cost, 2),
    }

    audit_trail = build_audit_trail(
        module="scheduling",
        solver_used="OR-Tools CP-SAT",
        status=status_str,
        solve_time_ms=solve_time_ms,
        num_variables=len(work) + len(total_shifts_vars),
        num_constraints=num_constraints,
        input_payload=input_payload,
        objective_name=s_cons.objective,
        objective_value=total_labor_cost if s_cons.objective == "minimize_cost" else float(max_sh - min_sh),
    )

    return {
        "status": status_str,
        "solver_used": "OR-Tools CP-SAT",
        "solve_time_ms": round(solve_time_ms, 2),
        "summary": summary,
        "schedule": schedule,
        "staff_workload": staff_workload,
        "diagnostics": None,
        "audit_trail": audit_trail,
    }
