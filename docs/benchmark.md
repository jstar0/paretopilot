# Benchmark contract

Run:

```bash
uv run paretopilot benchmark --output-dir /tmp/paretopilot-a
uv run paretopilot benchmark --output-dir /tmp/paretopilot-b
cmp /tmp/paretopilot-a/benchmark.json /tmp/paretopilot-b/benchmark.json
cmp /tmp/paretopilot-a/pareto_front.csv /tmp/paretopilot-b/pareto_front.csv
cmp /tmp/paretopilot-a/report.md /tmp/paretopilot-b/report.md
cmp /tmp/paretopilot-a/manifest.json /tmp/paretopilot-b/manifest.json
```

For seed stability:

```bash
uv run paretopilot sweep --output-dir /tmp/paretopilot-sweep-a --seeds 17,29,41
uv run paretopilot sweep --output-dir /tmp/paretopilot-sweep-b --seeds 17,29,41
cmp /tmp/paretopilot-sweep-a/sweep.json /tmp/paretopilot-sweep-b/sweep.json
cmp /tmp/paretopilot-sweep-a/manifest.json /tmp/paretopilot-sweep-b/manifest.json
```

The fixed seed and synthetic fixture make the state benchmark byte-stable. The
report records population, generations, feasible count, front size, baselines,
representative plans, and each plan's objective/constraint diagnostics. The
sweep report records per-seed representatives and population standard
deviations. It does not measure wall-clock performance and does not claim
real-world road or environmental outcomes.

Any future performance claim must state Python version, seed, population,
generation count, and machine; a single Pareto chart is not enough evidence.
