"""End-to-end MCP workflow tests (inspect_source -> dry_run_validate -> solve_*) and server registration."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from optima_mcp import server
from optima_mcp.server import (
    dry_run_validate,
    inspect_source,
    mcp,
    optima_strict_workflow,
    run_what_if_scenarios,
    solve_procurement,
    solve_routing,
    solve_scheduling,
    strict_rules_resource,
)

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


@pytest.mark.asyncio
async def test_mcp_server_registers_all_tools_resource_and_prompt() -> None:
    tools = await mcp.list_tools()
    tool_names = {t.name for t in tools}
    expected_tools = {
        "inspect_source",
        "dry_run_validate",
        "solve_procurement",
        "solve_routing",
        "solve_scheduling",
        "solve_allocation",
        "run_what_if_scenarios",
    }
    assert expected_tools == tool_names

    resources = await mcp.list_resources()
    resource_uris = {str(r.uri) for r in resources}
    assert "optima://protocol/strict-rules" in resource_uris

    rules_text = strict_rules_resource()
    assert "STRICT EXECUTION PROTOCOL (ZERO HALLUCINATION)" in rules_text
    assert "dry_run_validate" in rules_text

    prompts = await mcp.list_prompts()
    prompt_names = {p.name for p in prompts}
    assert "optima_strict_workflow" in prompt_names

    rendered_prompt = optima_strict_workflow("Optimize station fuel purchasing")
    assert "inspect_source" in rendered_prompt
    assert "Optimize station fuel purchasing" in rendered_prompt


def test_e2e_three_step_procurement_pipeline() -> None:
    source = {"path": str(EXAMPLES_DIR / "fuel_and_oils_station.csv")}

    # STEP 1: inspect_source (schema + max 3 sample rows only)
    inspection = inspect_source(source)
    assert inspection["row_count"] == 26
    assert len(inspection["sample_rows"]) == 3
    assert "sku_id" in inspection["columns"]
    assert "unit_cost" in inspection["numeric_summary"]

    # STEP 2: dry_run_validate (catches negative margin trap on PROMO-FRESH-1)
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
    constraints = {"max_budget": 50000.0, "max_storage_space": 1200.0}

    validation = dry_run_validate("procurement", source, mapping, constraints)
    assert validation["valid"] is True
    assert validation["ready_to_solve"] is True
    assert validation["valid_row_count"] == 26
    warning_codes = {w["code"] for w in validation["warnings"]}
    assert "NEGATIVE_MARGIN_TRAP" in warning_codes

    # STEP 3: solve_procurement (deterministic HiGHS MILP)
    solution = solve_procurement(source, mapping, constraints)
    assert solution["status"] == "Optimal"
    assert solution["summary"]["budget_utilization_pct"] <= 100.0
    assert solution["summary"]["storage_utilization_pct"] <= 100.0
    assert solution["audit_trail"]["deterministic"] is True
    assert len(solution["purchase_plan"]) > 0

    # STEP 3b: What-if sensitivity analysis
    what_if = run_what_if_scenarios(
        "procurement",
        source,
        mapping,
        constraints,
        [{"name": "Tighter Budget", "constraint_overrides": {"max_budget": 30000.0}}],
    )
    assert what_if["baseline"]["status"] == "Optimal"
    assert what_if["scenarios"][0]["status"] == "Optimal"


def test_dry_run_detects_storage_overflow_and_unit_variance(tmp_path: Path) -> None:
    df = pd.DataFrame(
        [
            {"id": "UNIT_A", "cost": 2.0, "price": 5.0, "demand": 50, "stock": 100, "space": 2.0},
            {"id": "UNIT_B", "cost": 3.0, "price": 6.0, "demand": 50, "stock": 100, "space": 2.0},
            # Possible carton/barrel entered next to single units (cost 300 vs median 3 -> 100x)
            {"id": "CARTON_C", "cost": 300.0, "price": 450.0, "demand": 10, "stock": 5, "space": 10.0},
        ]
    )
    path = tmp_path / "anomalies.csv"
    df.to_csv(path, index=False)

    report = dry_run_validate(
        "procurement",
        {"path": str(path)},
        {
            "item_id": "id",
            "cost": "cost",
            "selling_price": "price",
            "expected_demand": "demand",
            "current_stock": "stock",
            "unit_space": "space",
        },
        {"max_budget": 10000.0, "max_storage_space": 300.0},  # current space = 450 > 300!
    )
    codes = {w["code"] for w in report["warnings"]}
    assert "STORAGE_ALREADY_EXCEEDED" in codes
    assert "UNIT_PRICE_VARIANCE_HIGH" in codes


@pytest.mark.asyncio
async def test_mcp_call_tool_routing_and_scheduling_e2e() -> None:
    # Call directly through FastMCP's call_tool interface to verify JSON serialization end-to-end
    vrp_src = {"path": str(EXAMPLES_DIR / "fleet_delivery_orders.json")}
    vrp_map = {"stop_id": "stop_id", "lat": "lat", "lon": "lon", "demand": "demand_kg"}

    val_res = dry_run_validate("routing", vrp_src, vrp_map, {})
    assert val_res["valid"] is True

    raw_result = await mcp.call_tool(
        "solve_routing",
        {"source_config": vrp_src, "mapping": vrp_map, "constraints": {}},
    )
    # Support mcp 2.x CallToolResult (.structured_content / .content) and mcp 1.x tuples/lists
    if hasattr(raw_result, "structured_content") and raw_result.structured_content is not None:
        payload = raw_result.structured_content
    elif hasattr(raw_result, "content"):
        payload = json.loads(raw_result.content[0].text)
    else:
        blocks = raw_result[0] if isinstance(raw_result, tuple) else raw_result
        payload = json.loads(blocks[0].text)
    assert payload["status"] == "Optimal"
    assert payload["summary"]["stops_served"] == 12

    # Scheduling E2E
    sched_src = {"path": str(EXAMPLES_DIR / "staff_shifts_sample.csv")}
    sched_map = {
        "staff_id": "staff_id",
        "name": "full_name",
        "skills": "skills",
        "max_shifts": "max_shifts",
        "shift_cost": "shift_cost",
        "unavailable_days": "unavailable_days",
    }
    sched_cons = {"days": 7, "shifts": ["Morning", "Evening", "Night"], "required_headcount": 2}
    assert dry_run_validate("scheduling", sched_src, sched_map, sched_cons)["valid"] is True
    sched_res = solve_scheduling(sched_src, sched_map, sched_cons)
    assert sched_res["status"] in ("Optimal", "Feasible")
    assert hasattr(server, "main") and callable(server.main)
