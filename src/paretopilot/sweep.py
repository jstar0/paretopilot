"""Multi-seed robustness artifacts for ParetoPilot."""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import replace
from math import sqrt
from pathlib import Path
from statistics import fmean
from typing import Any

from .fixture import create_fixture
from .optimizer import NSGA2Config, nsga2
from .report import choose_representatives


def parse_seed_list(value: str) -> tuple[int, ...]:
    """Parse a comma-separated seed list while preserving user order."""

    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("seeds must be a comma-separated list of integers")
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except ValueError as exc:
        raise ValueError("seeds must be a comma-separated list of integers") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    return seeds


def seed_sweep_payload(
    seeds: Iterable[int], config: NSGA2Config | None = None
) -> dict[str, Any]:
    """Run the same synthetic scenario for several seeds.

    The aggregate is intentionally small and decision-oriented: it reports
    front size, feasible population count, and the four named representative
    plans.  It is not a statistical claim about a real maintenance network.
    """

    seed_tuple = tuple(seeds)
    if not seed_tuple:
        raise ValueError("at least one seed is required")
    if len(set(seed_tuple)) != len(seed_tuple):
        raise ValueError("seeds must be unique")
    if any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seed_tuple):
        raise ValueError("seeds must be integers")
    base_config = config or NSGA2Config()
    runs: list[dict[str, Any]] = []
    for seed in seed_tuple:
        result = nsga2(create_fixture(), replace(base_config, seed=seed))
        representatives = choose_representatives(result.pareto_front)
        balanced = representatives.get("balanced_topsis")
        if balanced is None:
            raise RuntimeError("optimizer returned no feasible representative")
        runs.append(
            {
                "seed": seed,
                "paretoFrontSize": len(result.pareto_front),
                "feasibleFinalPopulation": sum(
                    1 for member in result.final_population if member.evaluation.feasible
                ),
                "lowestCost": round(
                    representatives["lowest_cost"].evaluation.total_cost, 6
                ),
                "lowestEmissions": round(
                    representatives["lowest_emissions"].evaluation.total_emissions, 6
                ),
                "highestCondition": round(
                    representatives["highest_condition"].evaluation.final_mean_condition, 6
                ),
                "balancedCost": round(balanced.evaluation.total_cost, 6),
                "balancedEmissions": round(balanced.evaluation.total_emissions, 6),
                "balancedFinalMeanCondition": round(
                    balanced.evaluation.final_mean_condition, 6
                ),
                "balancedMinConditionSeen": round(
                    balanced.evaluation.min_condition_seen, 6
                ),
            }
        )
    aggregate: dict[str, Any] = {}
    for field in (
        "paretoFrontSize",
        "feasibleFinalPopulation",
        "lowestCost",
        "lowestEmissions",
        "highestCondition",
        "balancedCost",
        "balancedEmissions",
        "balancedFinalMeanCondition",
        "balancedMinConditionSeen",
    ):
        values = [float(run[field]) for run in runs]
        mean = fmean(values)
        variance = fmean((value - mean) ** 2 for value in values)
        aggregate[field] = {"mean": round(mean, 6), "pstdev": round(sqrt(variance), 6)}
    return {
        "sweepVersion": 1,
        "seeds": list(seed_tuple),
        "populationSize": base_config.population_size,
        "generations": base_config.generations,
        "crossoverRate": base_config.crossover_rate,
        "mutationRate": base_config.mutation_rate,
        "runs": runs,
        "aggregate": aggregate,
    }


def write_sweep(
    output_dir: str | Path, seeds: Iterable[int], config: NSGA2Config | None = None
) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    payload = seed_sweep_payload(seeds, config)
    write_text(directory / "sweep.json", json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_csv(directory / "sweep.csv", payload["runs"])
    write_text(directory / "report.md", markdown_sweep_report(payload))
    manifest = {
        "schemaVersion": 1,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256(path.read_bytes()),
            }
            for path in sorted(directory.glob("*"))
            if path.is_file() and path.name != "manifest.json"
        ],
    }
    write_text(directory / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return payload


def write_csv(path: Path, runs: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "paretoFrontSize",
        "feasibleFinalPopulation",
        "lowestCost",
        "lowestEmissions",
        "highestCondition",
        "balancedCost",
        "balancedEmissions",
        "balancedFinalMeanCondition",
        "balancedMinConditionSeen",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(runs)


def markdown_sweep_report(payload: dict[str, Any]) -> str:
    lines = [
        "# ParetoPilot seed sweep",
        "",
        "The same synthetic scenario and optimizer settings run across independent seeds.",
        "",
        f"- seeds: `{', '.join(str(seed) for seed in payload['seeds'])}`",
        f"- population / generations: `{payload['populationSize']} / {payload['generations']}`",
        "",
        "| seed | front | feasible population | balanced cost | balanced emissions | balanced final mean |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        lines.append(
            f"| {run['seed']} | {run['paretoFrontSize']} | {run['feasibleFinalPopulation']} | "
            f"{run['balancedCost']:.2f} | {run['balancedEmissions']:.2f} | "
            f"{run['balancedFinalMeanCondition']:.2f} |"
        )
    lines.extend(["", "## Aggregate", ""])
    for field, summary in payload["aggregate"].items():
        lines.append(f"- `{field}` — mean `{summary['mean']:.6f}`, population σ `{summary['pstdev']:.6f}`")
    lines.extend(
        [
            "",
            "This is a reproducibility/stability check over invented inputs, not a confidence interval "
            "or a claim about a real network.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
