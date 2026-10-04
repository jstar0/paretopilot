# ParetoPilot

ParetoPilot is a reproducible multi-objective planning lab. It turns a general maintenance brief into a synthetic scenario where an NSGA-II optimizer exposes the trade-off between intervention cost, emissions, and final asset condition.

## What makes it more than a toy

- **Constraint-domination:** infeasible plans are ranked by measurable violation before objective values; annual budgets and condition floors are never hidden behind a single score.
- **Real Pareto search:** deterministic seeding, crossover, mutation, non-dominated sorting, and crowding distance preserve diverse trade-off plans instead of returning one arbitrary weighted sum.
- **Decision layer:** lowest-cost, lowest-emissions, highest-condition, and weighted TOPSIS representatives make the front actionable without hiding the underlying trade-off.
- **Explainable state transition:** treatment boost occurs at the start of a year, decay depends on time since the last intervention, and every annual mean/minimum is retained for inspection.
- **Reproducible evidence:** the CLI writes JSON, CSV, Markdown, and a SHA-256 manifest; two runs with the same seed are byte-identical. A multi-seed sweep checks stability rather than presenting one lucky run.
- **Clean-room boundary:** the fixture contains invented segment IDs, costs, emissions, and condition curves. No client spreadsheets, locations, reports, or outputs are included.

## Quick start

```bash
uv sync --locked --dev
uv run pytest
uv run --with ruff ruff check src tests
uv lock --check
uv build
uv run paretopilot benchmark --output-dir /tmp/paretopilot-benchmark
uv run paretopilot sweep --output-dir /tmp/paretopilot-sweep --seeds 17,29,41
```

The benchmark accepts `--seed`, `--population`, and `--generations` for controlled experiments. It produces:

```text
benchmark.json    machine-readable scenario, front, and baselines
pareto_front.csv  one row per non-dominated plan
report.md         compact human-readable comparison
manifest.json     SHA-256 and byte sizes for every artifact
```

The `sweep` command writes `sweep.json`, `sweep.csv`, `report.md`, and a
second manifest. It repeats the same synthetic scenario across independent
seeds and reports population standard deviations for the representative
choices; it is a stability check, not a confidence interval.

## Model

Each plan is a `segments × years` matrix of treatment indices. Every year applies the selected treatment, advances natural decay, and records cost, emissions, and condition. The three minimization objectives are:

1. total intervention cost;
2. total emissions;
3. negative final mean condition (negative because NSGA-II minimizes all objectives).

The default synthetic scenario has 12 segments, six years, four treatment options, a yearly budget, and condition floors. The output deliberately includes baselines (`no_action`, `light_cycle`, and `performance_cycle`) and representative choices (`lowest_cost`, `lowest_emissions`, `highest_condition`, and `balanced_topsis`) so the front can be discussed rather than merely displayed. TOPSIS uses the three minimization objectives with default weights `(0.4, 0.3, 0.3)` and only ranks feasible front members.

## Evidence and limits

The benchmark is a software/reproducibility contract. Its values are not official road statistics, a real maintenance recommendation, or a claim of savings. See [`docs/architecture.md`](docs/architecture.md) and [`docs/benchmark.md`](docs/benchmark.md).

MIT licensed. Contributions must keep data synthetic or openly licensed, preserve deterministic seeds, and add tests for new constraints or operators.
