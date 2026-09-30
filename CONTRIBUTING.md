# Contributing to ParetoPilot

Keep the scenario synthetic, deterministic, and explainable.

```bash
uv sync --locked --dev
uv run pytest
uv run --with ruff ruff check src tests
uv lock --check
uv build
uv run paretopilot benchmark --output-dir /tmp/paretopilot-benchmark
```

New operators or constraints need tests for normal, infeasible, and repeated
seeded runs. Do not add client spreadsheets, locations, reports, credentials,
or unreviewed third-party algorithm implementations.
