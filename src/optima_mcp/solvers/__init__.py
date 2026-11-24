"""Deterministic Operations Research solvers (HiGHS MILP/LP & Google OR-Tools CP-SAT/VRP)."""

from optima_mcp.solvers.allocation import solve_allocation
from optima_mcp.solvers.procurement import get_milp_solver, solve_procurement
from optima_mcp.solvers.routing import haversine_km, solve_routing
from optima_mcp.solvers.scenarios import run_what_if_scenarios
from optima_mcp.solvers.scheduling import solve_scheduling

__all__ = [
    "get_milp_solver",
    "haversine_km",
    "run_what_if_scenarios",
    "solve_allocation",
    "solve_procurement",
    "solve_routing",
    "solve_scheduling",
]
