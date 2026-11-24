"""Deterministic Vehicle Routing Problem (VRP) Solver powered by Google OR-Tools."""

from __future__ import annotations

import math
import time
from typing import Any

import pandas as pd
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

from optima_mcp.connectors.base import load_dataframe
from optima_mcp.validation.audit import build_audit_trail, diagnose_routing_infeasibility
from optima_mcp.validation.schemas import (
    RoutingConstraints,
    RoutingMapping,
    RoutingStop,
    parse_routing_dataframe,
)

EARTH_RADIUS_KM = 6371.0088
DIST_SCALE = 1000  # Convert km -> integer meters for exact OR-Tools arc costs
DEMAND_SCALE = 100  # Preserve 2 decimal places on capacities & demands


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance in kilometers between two WGS84 coordinates."""
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_KM * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def _build_haversine_matrix(stops: list[RoutingStop]) -> list[list[float]]:
    n = len(stops)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_km(
                float(stops[i].lat),  # type: ignore[arg-type]
                float(stops[i].lon),  # type: ignore[arg-type]
                float(stops[j].lat),  # type: ignore[arg-type]
                float(stops[j].lon),  # type: ignore[arg-type]
            )
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def solve_routing(
    source_config: dict[str, Any],
    mapping: dict[str, Any],
    constraints: dict[str, Any],
    *,
    df_override: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Solve Capacitated Vehicle Routing Problem (CVRP) using Google OR-Tools."""
    t0 = time.perf_counter()

    r_map = RoutingMapping.model_validate(mapping)
    r_cons = RoutingConstraints.model_validate(constraints)
    df = df_override if df_override is not None else load_dataframe(source_config)

    stops, resolved_cons, parse_errors = parse_routing_dataframe(
        df, r_map, r_cons, source_config=source_config
    )
    if parse_errors or not stops:
        return {
            "status": "ValidationError",
            "errors": parse_errors or ["No valid routing stops found."],
            "solver_used": None,
            "solve_time_ms": round((time.perf_counter() - t0) * 1000, 2),
        }

    # Resolve depot & node list
    all_nodes: list[RoutingStop] = list(stops)
    depot_idx: int

    flagged_depots = [i for i, s in enumerate(all_nodes) if s.is_depot]
    if resolved_cons.distance_matrix is not None:
        dist_matrix_km = resolved_cons.distance_matrix
        if len(dist_matrix_km) != len(all_nodes):
            return {
                "status": "ValidationError",
                "errors": [
                    f"distance_matrix dimension ({len(dist_matrix_km)}) does not match "
                    f"number of stops ({len(all_nodes)})."
                ],
                "solver_used": None,
                "solve_time_ms": round((time.perf_counter() - t0) * 1000, 2),
            }
        depot_idx = flagged_depots[0] if flagged_depots else resolved_cons.depot_index
    else:
        if flagged_depots:
            depot_idx = flagged_depots[0]
        elif resolved_cons.depot is not None:
            depot_stop = RoutingStop(
                stop_id=str(resolved_cons.depot.get("stop_id", "DEPOT")),
                lat=float(resolved_cons.depot["lat"]),
                lon=float(resolved_cons.depot["lon"]),
                demand=0.0,
                is_depot=True,
            )
            all_nodes = [depot_stop] + all_nodes
            depot_idx = 0
        else:
            depot_idx = resolved_cons.depot_index
            all_nodes[depot_idx] = all_nodes[depot_idx].model_copy(update={"is_depot": True, "demand": 0.0})

        dist_matrix_km = _build_haversine_matrix(all_nodes)

    num_nodes = len(all_nodes)
    num_vehicles = int(resolved_cons.num_vehicles or 1)
    capacities = list(resolved_cons.vehicle_capacities or [])

    # Quick pre-check when dropping stops is disallowed
    non_depot_demand = sum(s.demand for i, s in enumerate(all_nodes) if i != depot_idx)
    if not resolved_cons.allow_drop_stops and (
        non_depot_demand > sum(capacities) + 1e-6
        or any(s.demand > max(capacities) + 1e-6 for i, s in enumerate(all_nodes) if i != depot_idx)
    ):
        solve_time_ms = (time.perf_counter() - t0) * 1000.0
        diagnostics = diagnose_routing_infeasibility(all_nodes, resolved_cons, dist_matrix_km, depot_idx)
        audit_trail = build_audit_trail(
            module="routing",
            solver_used="OR-Tools Routing",
            status="Infeasible",
            solve_time_ms=solve_time_ms,
            num_variables=num_nodes * num_vehicles,
            num_constraints=num_nodes + num_vehicles,
            input_payload={
                "module": "routing",
                "constraints": resolved_cons.model_dump(exclude={"distance_matrix"}),
                "stops": [s.model_dump() for s in all_nodes],
            },
            objective_name="minimize_total_distance",
        )
        return {
            "status": "Infeasible",
            "solver_used": "OR-Tools Routing",
            "solve_time_ms": round(solve_time_ms, 2),
            "summary": {},
            "routes": [],
            "dropped_stops": [],
            "diagnostics": diagnostics,
            "audit_trail": audit_trail,
        }

    # Build OR-Tools RoutingModel
    int_dist = [[int(round(d * DIST_SCALE)) for d in row] for row in dist_matrix_km]
    int_demands = [
        0 if i == depot_idx else int(round(s.demand * DEMAND_SCALE)) for i, s in enumerate(all_nodes)
    ]
    int_caps = [int(round(c * DEMAND_SCALE)) for c in capacities]

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_idx)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index: int, to_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int_dist[from_node][to_node]

    transit_cb_idx = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_cb_idx)

    # Capacity dimension
    def demand_callback(from_index: int) -> int:
        from_node = manager.IndexToNode(from_index)
        return int_demands[from_node]

    demand_cb_idx = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_cb_idx,
        0,  # null capacity slack
        int_caps,
        True,  # start cumul to zero
        "Capacity",
    )

    # Optional max route distance dimension
    if resolved_cons.max_route_distance is not None:
        max_dist_scaled = int(round(resolved_cons.max_route_distance * DIST_SCALE))
        routing.AddDimension(
            transit_cb_idx,
            0,
            max_dist_scaled,
            True,
            "Distance",
        )

    # Optional stop dropping with huge penalty so solver drops minimum possible stops
    if resolved_cons.allow_drop_stops:
        max_arc = max((max(row) for row in int_dist), default=1000)
        drop_penalty = max(10_000_000, max_arc * num_nodes * 10)
        for node in range(num_nodes):
            if node == depot_idx:
                continue
            routing.AddDisjunction([manager.NodeToIndex(node)], drop_penalty)

    search_params = pywrapcp.DefaultRoutingSearchParameters()
    search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_params.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_params.time_limit.FromSeconds(min(resolved_cons.time_limit_seconds, 5))
    # Cap solution search nicely for tiny instances while still getting great routes
    if num_nodes <= 15:
        search_params.time_limit.FromMilliseconds(500)

    solution = routing.SolveWithParameters(search_params)
    solve_time_ms = (time.perf_counter() - t0) * 1000.0

    input_payload = {
        "module": "routing",
        "constraints": resolved_cons.model_dump(exclude={"distance_matrix"}),
        "stops": [s.model_dump() for s in all_nodes],
    }

    if solution is None:
        diagnostics = diagnose_routing_infeasibility(all_nodes, resolved_cons, dist_matrix_km, depot_idx)
        audit_trail = build_audit_trail(
            module="routing",
            solver_used="OR-Tools Routing",
            status="Infeasible",
            solve_time_ms=solve_time_ms,
            num_variables=num_nodes * num_vehicles,
            num_constraints=num_nodes + num_vehicles,
            input_payload=input_payload,
            objective_name="minimize_total_distance",
        )
        return {
            "status": "Infeasible",
            "solver_used": "OR-Tools Routing",
            "solve_time_ms": round(solve_time_ms, 2),
            "summary": {},
            "routes": [],
            "dropped_stops": [],
            "diagnostics": diagnostics,
            "audit_trail": audit_trail,
        }

    # Extract routes & dropped stops
    routes: list[dict[str, Any]] = []
    visited_nodes: set[int] = {depot_idx}
    total_distance_km = 0.0
    total_demand_served = 0.0
    vehicles_used = 0

    for v_idx in range(num_vehicles):
        index = routing.Start(v_idx)
        route_stops: list[dict[str, Any]] = []
        route_dist_km = 0.0
        route_load = 0.0

        while not routing.IsEnd(index):
            node = manager.IndexToNode(index)
            stop = all_nodes[node]
            if node != depot_idx:
                visited_nodes.add(node)
                route_load += stop.demand
            route_stops.append(
                {
                    "stop_id": stop.stop_id,
                    "is_depot": node == depot_idx,
                    "demand": round(stop.demand, 2),
                    "cumulative_distance_km": round(route_dist_km, 3),
                }
            )
            next_index = solution.Value(routing.NextVar(index))
            next_node = manager.IndexToNode(next_index)
            route_dist_km += dist_matrix_km[node][next_node]
            index = next_index

        # Append return to depot
        end_node = manager.IndexToNode(index)
        route_stops.append(
            {
                "stop_id": all_nodes[end_node].stop_id,
                "is_depot": True,
                "demand": 0.0,
                "cumulative_distance_km": round(route_dist_km, 3),
            }
        )

        # Only count vehicle as used if it visited at least one non-depot stop
        stops_visited_count = len(route_stops) - 2
        if stops_visited_count > 0:
            vehicles_used += 1
        total_distance_km += route_dist_km
        total_demand_served += route_load

        cap = capacities[v_idx]
        load_util_pct = (route_load / cap * 100.0) if cap > 0 else 0.0
        routes.append(
            {
                "vehicle_id": f"V{v_idx + 1}",
                "stops_visited_count": stops_visited_count,
                "stops": route_stops,
                "route_distance_km": round(route_dist_km, 3),
                "route_load": round(route_load, 2),
                "vehicle_capacity": round(cap, 2),
                "load_utilization_pct": round(load_util_pct, 2),
            }
        )

    dropped_stops = [
        {
            "stop_id": all_nodes[i].stop_id,
            "demand": round(all_nodes[i].demand, 2),
            "reason": "Dropped due to vehicle capacity or max_route_distance constraints.",
        }
        for i in range(num_nodes)
        if i not in visited_nodes
    ]

    total_fleet_cap = sum(capacities)
    fleet_util_pct = (total_demand_served / total_fleet_cap * 100.0) if total_fleet_cap > 0 else 0.0
    total_customer_stops = num_nodes - 1

    summary = {
        "total_distance_km": round(total_distance_km, 3),
        "vehicles_available": num_vehicles,
        "vehicles_used": vehicles_used,
        "stops_total": total_customer_stops,
        "stops_served": total_customer_stops - len(dropped_stops),
        "stops_dropped": len(dropped_stops),
        "total_demand_served": round(total_demand_served, 2),
        "total_fleet_capacity": round(total_fleet_cap, 2),
        "fleet_load_utilization_pct": round(fleet_util_pct, 2),
    }

    audit_trail = build_audit_trail(
        module="routing",
        solver_used="OR-Tools Routing",
        status="Optimal",
        solve_time_ms=solve_time_ms,
        num_variables=num_nodes * num_vehicles,
        num_constraints=num_nodes + num_vehicles,
        input_payload=input_payload,
        objective_name="minimize_total_distance",
        objective_value=total_distance_km,
    )

    return {
        "status": "Optimal",
        "solver_used": "OR-Tools Routing",
        "solve_time_ms": round(solve_time_ms, 2),
        "summary": summary,
        "routes": routes,
        "dropped_stops": dropped_stops,
        "diagnostics": None,
        "audit_trail": audit_trail,
    }
