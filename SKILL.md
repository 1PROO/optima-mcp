---
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
