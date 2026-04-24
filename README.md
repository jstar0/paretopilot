# ParetoPilot

ParetoPilot is a reproducible multi-objective planning lab. It turns a general maintenance brief into a synthetic scenario where an NSGA-II optimizer exposes the trade-off between intervention cost, emissions, and final asset condition.

## What makes it more than a toy

- **Constraint-domination:** infeasible plans are ranked by measurable violation before objective values; annual budgets and condition floors are never hidden behind a single score.
- **Real Pareto search:** deterministic seeding, crossover, mutation, non-dominated sorting, and crowding distance preserve diverse trade-off plans instead of returning one arbitrary weighted sum.
- **Decision layer:** lowest-cost, lowest-emissions, highest-condition, and weighted TOPSIS representatives make the front actionable without hiding the underlying trade-off.
- **Explainable state transition:** treatment boost occurs at the start of a year, decay depends on time since the last intervention, and every annual mean/minimum is retained for inspection.
- **Reproducible evidence:** the CLI writes JSON, CSV, Markdown, and a SHA-256 manifest; two runs with the same seed are byte-identical. A multi-seed sweep checks stability rather than presenting one lucky run.
- **Clean-room boundary:** the fixture contains invented segment IDs, costs, emissions, and condition curves. No client spreadsheets, locations, reports, or outputs are included.
