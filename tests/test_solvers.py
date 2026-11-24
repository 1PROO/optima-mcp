"""Mathematical optimality & infeasibility tests for all 4 solvers + What-If scenarios."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from optima_mcp.solvers.allocation import solve_allocation
from optima_mcp.solvers.procurement import solve_procurement
from optima_mcp.solvers.routing import haversine_km, solve_routing
from optima_mcp.solvers.scenarios import run_what_if_scenarios
from optima_mcp.solvers.scheduling import solve_scheduling

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


# ---------------------------------------------------------------------------
# 1. Procurement Solver (HiGHS MILP)
# ---------------------------------------------------------------------------


def test_procurement_exact_optimality_and_shipping_moq(tmp_path: Path) -> None:
    """Hand-verifiable MILP instance with budget, MOQ, fixed shipping, and negative-margin trap."""
    df = pd.DataFrame(
        [
            # High margin: profit = +10/unit, cost = 10, need = 10 -> max profit = 100
            {
                "id": "HIGH",
                "cost": 10.0,
                "price": 20.0,
                "demand": 10,
                "stock": 0,
                "moq": 5,
                "space": 1.0,
                "sup": "S1",
                "ship": 20.0,
            },
            # Medium margin: profit = +3/unit, cost = 10, need = 10 -> max profit = 30
            {
                "id": "MED",
                "cost": 10.0,
                "price": 13.0,
                "demand": 10,
                "stock": 0,
                "moq": 2,
                "space": 1.0,
                "sup": "S1",
                "ship": 20.0,
            },
            # Negative margin trap: cost >= price -> must be skipped
            {
                "id": "LOSS",
                "cost": 15.0,
                "price": 12.0,
                "demand": 10,
                "stock": 0,
                "moq": 0,
                "space": 1.0,
                "sup": "S2",
                "ship": 0.0,
            },
        ]
    )
    path = tmp_path / "proc.csv"
    df.to_csv(path, index=False)

    mapping = {
        "item_id": "id",
        "cost": "cost",
        "selling_price": "price",
        "expected_demand": "demand",
        "current_stock": "stock",
        "min_order_qty": "moq",
        "unit_space": "space",
        "supplier_id": "sup",
        "supplier_fixed_shipping_cost": "ship",
    }
    # Budget = 150: Buy 10 HIGH (100) + shipping S1 (20) + 3 MED (30) = 150 exact spend!
    # Profit = 10*10 + 3*3 - 20 = 89.0 exact!
    res = solve_procurement(
        {"path": str(path)},
        mapping,
        {"max_budget": 150.0, "max_storage_space": 20.0, "objective": "maximize_profit"},
    )
    assert res["status"] == "Optimal"
    assert res["solver_used"] in ("HiGHS", "CBC")
    assert res["summary"]["total_spend"] == pytest.approx(150.0)
    assert res["summary"]["total_shipping_cost"] == pytest.approx(20.0)
    assert res["summary"]["expected_profit"] == pytest.approx(89.0)

    qtys = {r["item_id"]: r["order_qty"] for r in res["purchase_plan"]}
    assert qtys == {"HIGH": 10, "MED": 3}
    skipped_ids = {r["item_id"]: r["reason"] for r in res["skipped_opportunities"]}
    assert "LOSS" in skipped_ids
    assert "margin" in skipped_ids["LOSS"].lower()


def test_procurement_maximize_roi_and_minimize_cost(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            # ROI = (30 - 10) / 10 = 200%
            {"id": "A", "cost": 10.0, "price": 30.0, "demand": 5, "stock": 0},
            # ROI = (15 - 10) / 10 = 50%
            {"id": "B", "cost": 10.0, "price": 15.0, "demand": 20, "stock": 0},
        ]
    )
    path = tmp_path / "roi.csv"
    df.to_csv(path, index=False)
    mapping = {
        "item_id": "id",
        "cost": "cost",
        "selling_price": "price",
        "expected_demand": "demand",
        "current_stock": "stock",
    }

    # Maximize ROI must buy ONLY A (200% ROI) rather than diluting with B (50% ROI)
    res_roi = solve_procurement(
        {"path": str(path)},
        mapping,
        {"max_budget": 500.0, "objective": "maximize_roi"},
    )
    assert res_roi["status"] == "Optimal"
    assert res_roi["summary"]["roi_pct"] == pytest.approx(200.0, abs=0.01)
    assert all(r["item_id"] == "A" for r in res_roi["purchase_plan"])

    # Minimize cost with 80% coverage of 25 total demand = 20 units * $10 = $200
    res_min = solve_procurement(
        {"path": str(path)},
        mapping,
        {"max_budget": 500.0, "min_demand_coverage": 0.8, "objective": "minimize_cost"},
    )
    assert res_min["status"] == "Optimal"
    assert res_min["summary"]["total_spend"] == pytest.approx(200.0)
    assert res_min["summary"]["demand_coverage_pct"] >= 80.0


def test_procurement_infeasibility_diagnostics() -> None:
    mapping = {
        "item_id": "sku_id",
        "cost": "unit_cost",
        "selling_price": "selling_price",
        "expected_demand": "expected_demand",
        "current_stock": "current_stock",
        "unit_space": "unit_space",
        "min_order_qty": "min_order_qty",
        "max_order_qty": "max_order_qty",
        "shelf_life_days": "shelf_life_days",
        "supplier_id": "supplier_id",
        "supplier_fixed_shipping_cost": "supplier_fixed_shipping_cost",
    }
    csv_path = str(EXAMPLES_DIR / "fuel_and_oils_station.csv")

    # 1. Realistic README example solves cleanly
    res_ok = solve_procurement(
        {"path": csv_path},
        mapping,
        {"max_budget": 50000.0, "max_storage_space": 1200.0, "objective": "maximize_profit"},
    )
    assert res_ok["status"] == "Optimal"
    assert res_ok["summary"]["total_spend"] <= 50000.0 + 1e-6
    assert res_ok["summary"]["storage_used"] <= 1200.0 + 1e-6
    assert res_ok["summary"]["expected_profit"] > 0
    # Negative-margin promo item must be skipped
    skipped_ids = {s["item_id"] for s in res_ok["skipped_opportunities"]}
    assert "PROMO-FRESH-1" in skipped_ids

    # 2. Infeasible: impossible budget for 95% coverage + tiny storage cap
    res_inf = solve_procurement(
        {"path": csv_path},
        mapping,
        {
            "max_budget": 500.0,
            "max_storage_space": 100.0,
            "min_demand_coverage": 0.95,
            "objective": "maximize_profit",
        },
    )
    assert res_inf["status"] == "Infeasible"
    assert res_inf["diagnostics"] is not None
    assert res_inf["diagnostics"]["feasible"] is False
    assert len(res_inf["diagnostics"]["conflicting_constraints"]) >= 2


# ---------------------------------------------------------------------------
# 2. Vehicle Routing Solver (Google OR-Tools VRP)
# ---------------------------------------------------------------------------


def test_routing_exact_distance_matrix_and_example_vrp(tmp_path: Path) -> None:
    # 4-node square (side=10km): Depot(0) -> A(1) -> B(2) -> C(3) -> Depot(0) = 40km exact
    df = pd.DataFrame(
        [
            {"id": "DEPOT", "demand": 0, "is_depot": True},
            {"id": "A", "demand": 10, "is_depot": False},
            {"id": "B", "demand": 10, "is_depot": False},
            {"id": "C", "demand": 10, "is_depot": False},
        ]
    )
    path = tmp_path / "square.csv"
    df.to_csv(path, index=False)

    dist = [
        [0.0, 10.0, 14.142, 10.0],
        [10.0, 0.0, 10.0, 14.142],
        [14.142, 10.0, 0.0, 10.0],
        [10.0, 14.142, 10.0, 0.0],
    ]
    res_sq = solve_routing(
        {"path": str(path)},
        {"stop_id": "id", "demand": "demand", "is_depot": "is_depot"},
        {"num_vehicles": 1, "vehicle_capacities": [30], "distance_matrix": dist},
    )
    assert res_sq["status"] == "Optimal"
    assert res_sq["summary"]["total_distance_km"] == pytest.approx(40.0, abs=0.01)
    assert res_sq["summary"]["stops_served"] == 3
    assert res_sq["dropped_stops"] == []

    # Example fleet delivery JSON with Haversine km matrix
    assert haversine_km(30.0444, 31.2357, 30.0444, 31.2357) == pytest.approx(0.0)
    res_ex = solve_routing(
        {"path": str(EXAMPLES_DIR / "fleet_delivery_orders.json")},
        {"stop_id": "stop_id", "lat": "lat", "lon": "lon", "demand": "demand_kg"},
        {},  # Auto-reads depot & 3 vehicles from JSON
    )
    assert res_ex["status"] == "Optimal"
    assert res_ex["summary"]["stops_served"] == 12
    assert res_ex["summary"]["stops_dropped"] == 0
    for route in res_ex["routes"]:
        assert route["route_load"] <= route["vehicle_capacity"] + 1e-6


def test_routing_infeasibility_and_drop_stops() -> None:
    vrp_path = str(EXAMPLES_DIR / "fleet_delivery_orders.json")
    mapping = {"stop_id": "stop_id", "lat": "lat", "lon": "lon", "demand": "demand_kg"}

    # Infeasible when total capacity (300) < total demand (1480) and allow_drop_stops=False
    res_inf = solve_routing(
        {"path": vrp_path},
        mapping,
        {"num_vehicles": 2, "vehicle_capacities": [150, 150], "allow_drop_stops": False},
    )
    assert res_inf["status"] == "Infeasible"
    assert res_inf["diagnostics"] is not None
    assert any(
        c["constraint"] == "vehicle_capacities"
        for c in res_inf["diagnostics"]["conflicting_constraints"]
    )

    # With allow_drop_stops=True, serves subset within capacity and reports dropped_stops
    res_drop = solve_routing(
        {"path": vrp_path},
        mapping,
        {"num_vehicles": 2, "vehicle_capacities": [250, 250], "allow_drop_stops": True},
    )
    assert res_drop["status"] == "Optimal"
    assert res_drop["summary"]["stops_served"] > 0
    assert res_drop["summary"]["stops_dropped"] == len(res_drop["dropped_stops"]) > 0
    for r in res_drop["routes"]:
        assert r["route_load"] <= r["vehicle_capacity"] + 1e-6


# ---------------------------------------------------------------------------
# 3. Workforce Scheduling Solver (Google OR-Tools CP-SAT)
# ---------------------------------------------------------------------------


def test_scheduling_optimality_and_constraints_verification() -> None:
    csv_path = str(EXAMPLES_DIR / "staff_shifts_sample.csv")
    mapping = {
        "staff_id": "staff_id",
        "name": "full_name",
        "skills": "skills",
        "max_shifts": "max_shifts",
        "shift_cost": "shift_cost",
        "unavailable_days": "unavailable_days",
    }
    constraints = {
        "days": 7,
        "shifts": ["Morning", "Evening", "Night"],
        "required_headcount": 2,
        "required_skills": {"Supervisor": 1},
        "no_back_to_back": True,
        "max_shifts_per_day": 1,
        "objective": "balance_workload",
    }
    res = solve_scheduling({"path": csv_path}, mapping, constraints)
    assert res["status"] in ("Optimal", "Feasible")
    assert res["solver_used"] == "OR-Tools CP-SAT"
    assert res["summary"]["total_shifts_filled"] == 7 * 3 * 2  # 42 slots

    # Load staff lookup to verify EVERY constraint deterministically
    df = pd.read_csv(csv_path)
    staff_skills = {
        row["staff_id"]: set(str(row["skills"]).split(";")) for _, row in df.iterrows()
    }
    staff_unavail = {
        row["staff_id"]: (
            set(str(row["unavailable_days"]).split(";"))
            if pd.notna(row["unavailable_days"]) and str(row["unavailable_days"]).strip()
            else set()
        )
        for _, row in df.iterrows()
    }

    days = [f"Day_{i + 1}" for i in range(7)]
    shifts = ["Morning", "Evening", "Night"]

    # Check headcount, required Supervisor skill, and unavailability
    timeline_assignments: dict[str, list[int]] = {sid: [] for sid in staff_skills}
    t_idx = 0
    for d in days:
        day_seen: dict[str, int] = {}
        for sh in shifts:
            assigned = res["schedule"][d][sh]
            assert len(assigned) == 2
            assert any("Supervisor" in staff_skills[sid] for sid in assigned)
            for sid in assigned:
                assert d not in staff_unavail[sid]
                day_seen[sid] = day_seen.get(sid, 0) + 1
                timeline_assignments[sid].append(t_idx)
            t_idx += 1
        assert all(cnt <= 1 for cnt in day_seen.values())

    # Check max_shifts and NO back-to-back shifts across entire week (including Night -> next Morning)
    for sid, slots in timeline_assignments.items():
        assert len(slots) <= 5
        for a, b in zip(slots, slots[1:]):
            assert b - a >= 2, f"Back-to-back violation for {sid}: slots {a} and {b}"


def test_scheduling_infeasibility_diagnostics() -> None:
    csv_path = str(EXAMPLES_DIR / "staff_shifts_sample.csv")
    mapping = {
        "staff_id": "staff_id",
        "skills": "skills",
        "max_shifts": "max_shifts",
    }
    # Require 4 staff per shift * 21 shifts = 84 slots > 50 total staff max_shifts!
    res_inf = solve_scheduling(
        {"path": csv_path},
        mapping,
        {
            "days": 7,
            "shifts": ["Morning", "Evening", "Night"],
            "required_headcount": 4,
        },
    )
    assert res_inf["status"] == "Infeasible"
    assert res_inf["diagnostics"] is not None
    assert len(res_inf["diagnostics"]["conflicting_constraints"]) >= 1


# ---------------------------------------------------------------------------
# 4. Allocation Solver (HiGHS LP / MILP)
# ---------------------------------------------------------------------------


def test_allocation_lp_milp_and_infeasibility(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"id": "ASSET_A", "ret": 0.18, "cost": 1.0, "min": 100.0, "max": 600.0, "cat": "Growth"},
            {"id": "ASSET_B", "ret": 0.12, "cost": 1.0, "min": 0.0, "max": 800.0, "cat": "Growth"},
            {"id": "ASSET_C", "ret": 0.07, "cost": 1.0, "min": 0.0, "max": 1000.0, "cat": "Safe"},
        ]
    )
    path = tmp_path / "alloc.csv"
    df.to_csv(path, index=False)
    mapping = {
        "item_id": "id",
        "expected_return": "ret",
        "unit_cost": "cost",
        "min_allocation": "min",
        "max_allocation": "max",
        "category": "cat",
    }

    # Total budget = 1000, Safe >= 30% (300), Growth <= 70% (700)
    # Optimal: ASSET_A = 600 (cap), ASSET_B = 100 (remaining Growth), ASSET_C = 300 (min Safe)
    # Return = 600*0.18 + 100*0.12 + 300*0.07 = 108 + 12 + 21 = 141.0 exact!
    res = solve_allocation(
        {"path": str(path)},
        mapping,
        {
            "total_budget": 1000.0,
            "category_quotas": {"Safe": {"min_pct": 0.30}, "Growth": {"max_pct": 0.70}},
        },
    )
    assert res["status"] == "Optimal"
    assert res["summary"]["total_expected_return"] == pytest.approx(141.0)
    alloc_map = {a["item_id"]: a["allocated_budget"] for a in res["allocations"]}
    assert alloc_map == {"ASSET_A": 600.0, "ASSET_B": 100.0, "ASSET_C": 300.0}

    # Infeasible when Safe min_pct (0.95 -> 950) conflicts with ASSET_A min (100) under 1000 budget
    res_inf = solve_allocation(
        {"path": str(path)},
        mapping,
        {
            "total_budget": 1000.0,
            "category_quotas": {"Safe": {"min_pct": 0.95}},
        },
    )
    assert res_inf["status"] == "Infeasible"
    assert res_inf["diagnostics"] is not None


# ---------------------------------------------------------------------------
# 5. What-If Sensitivity Scenarios
# ---------------------------------------------------------------------------


def test_what_if_scenarios_procurement_and_limit() -> None:
    csv_path = str(EXAMPLES_DIR / "fuel_and_oils_station.csv")
    mapping = {
        "item_id": "sku_id",
        "cost": "unit_cost",
        "selling_price": "selling_price",
        "expected_demand": "expected_demand",
        "current_stock": "current_stock",
        "unit_space": "unit_space",
        "min_order_qty": "min_order_qty",
        "max_order_qty": "max_order_qty",
        "supplier_id": "supplier_id",
        "supplier_fixed_shipping_cost": "supplier_fixed_shipping_cost",
    }
    base_constraints = {"max_budget": 50000.0, "max_storage_space": 1200.0}
    scenarios = [
        {"name": "Budget Cut -20%", "constraint_overrides": {"max_budget": 40000.0}},
        {"name": "Supplier Cost +8%", "data_multipliers": {"cost": 1.08}},
        {"name": "Expanded Storage", "constraint_overrides": {"max_storage_space": 1800.0}},
    ]

    result = run_what_if_scenarios("procurement", {"path": csv_path}, mapping, base_constraints, scenarios)
    assert result["baseline"]["status"] == "Optimal"
    assert len(result["scenarios"]) == 3
    assert len(result["comparison_table"]) == 4

    base_profit = result["baseline"]["summary"]["expected_profit"]
    by_name = {s["name"]: s for s in result["scenarios"]}
    # Cost +8% must reduce profit relative to baseline
    assert by_name["Supplier Cost +8%"]["deltas_vs_baseline"]["expected_profit"]["delta"] < 0
    # Expanded storage must yield >= baseline profit
    assert by_name["Expanded Storage"]["summary"]["expected_profit"] >= base_profit - 1e-6

    # Reject > 5 scenarios
    with pytest.raises(ValueError, match="Maximum allowed"):
        run_what_if_scenarios(
            "procurement",
            {"path": csv_path},
            mapping,
            base_constraints,
            [{"name": f"S{i}"} for i in range(6)],
        )
