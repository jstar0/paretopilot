"""Stable JSON/CSV/Markdown artifacts for review."""

from __future__ import annotations

import csv
import hashlib
import json
from math import isfinite, sqrt
from pathlib import Path
from typing import Any

from .fixture import create_fixture
from .model import Evaluation, Plan, Scenario, evaluate_plan, plan_to_dict
from .optimizer import Individual, NSGA2Config, OptimizationResult, nsga2, objective_sort_key


def build_baselines(scenario: Scenario) -> dict[str, Plan]:
    no_action = Plan.no_action(scenario)
    light_rows = [[1 if year % 3 == 0 else 0 for year in range(scenario.years)] for _ in scenario.segments]
    performance_rows = [[3 if year % 3 == 0 else 0 for year in range(scenario.years)] for _ in scenario.segments]
    return {
        "no_action": no_action,
        "light_cycle": Plan.from_rows(light_rows, scenario),
        "performance_cycle": Plan.from_rows(performance_rows, scenario),
    }


def benchmark_payload(result: OptimizationResult) -> dict[str, Any]:
    front = result.pareto_front
    baseline_payload = {
        name: evaluation_payload(evaluate_plan(plan, result.scenario))
        for name, plan in build_baselines(result.scenario).items()
    }
    feasible_count = sum(1 for member in result.final_population if member.evaluation.feasible)
    representatives = choose_representatives(front)
    balanced_scores = {id(member): score for score, member in topsis_rank(front)}
    return {
        "benchmarkVersion": 1,
        "seed": result.config.seed,
        "populationSize": result.config.population_size,
        "generations": result.config.generations,
        "scenario": {
            "segments": len(result.scenario.segments),
            "years": result.scenario.years,
            "treatments": len(result.scenario.treatments),
            "annualBudget": result.scenario.annual_budget,
            "minCondition": result.scenario.min_condition,
            "minFinalMean": result.scenario.min_final_mean,
        },
        "finalPopulationSize": len(result.final_population),
        "feasibleFinalPopulation": feasible_count,
        "paretoFrontSize": len(front),
        "generationFrontSizes": list(result.generation_front_sizes),
        "front": [individual_payload(member) for member in front],
        "representatives": {label: individual_payload(member) for label, member in representatives.items()},
        "representativeScores": {
            label: round(balanced_scores.get(id(member), 0.0), 6)
            for label, member in representatives.items()
        },
        "baselines": baseline_payload,
    }


def write_benchmark(output_dir: str | Path, config: NSGA2Config | None = None) -> dict[str, Any]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    result = nsga2(create_fixture(), config or NSGA2Config())
    payload = benchmark_payload(result)
    json_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    write_text(directory / "benchmark.json", json_text)
    write_front_csv(directory / "pareto_front.csv", result.pareto_front)
    write_text(directory / "report.md", markdown_report(payload))
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


def evaluation_payload(evaluation: Evaluation) -> dict[str, Any]:
    return {
        "cost": round(evaluation.total_cost, 6),
        "emissions": round(evaluation.total_emissions, 6),
        "finalMeanCondition": round(evaluation.final_mean_condition, 6),
        "minConditionSeen": round(evaluation.min_condition_seen, 6),
        "feasible": evaluation.feasible,
        "violation": round(evaluation.violation, 6),
    }


def individual_payload(individual: Individual) -> dict[str, Any]:
    return {
        "objectives": [round(value, 6) for value in individual.evaluation.objectives],
        "cost": round(individual.evaluation.total_cost, 6),
        "emissions": round(individual.evaluation.total_emissions, 6),
        "finalMeanCondition": round(individual.evaluation.final_mean_condition, 6),
        "minConditionSeen": round(individual.evaluation.min_condition_seen, 6),
        "feasible": individual.evaluation.feasible,
        "violation": round(individual.evaluation.violation, 6),
        "plan": plan_to_dict(individual.plan),
    }


def topsis_rank(
    individuals: tuple[Individual, ...], weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
) -> list[tuple[float, Individual]]:
    """Rank feasible front members with a transparent TOPSIS score.

    All objectives are minimization objectives.  The returned score is higher
    when a plan is closer to the weighted ideal and farther from the anti-ideal.
    """

    if len(weights) != 3 or any(not isfinite(weight) or weight < 0 for weight in weights) or sum(weights) <= 0:
        raise ValueError("TOPSIS weights must be three non-negative values with a positive sum")
    front = [member for member in individuals if member.evaluation.feasible]
    if not front:
        return []
    columns = tuple(zip(*(member.evaluation.objectives for member in front), strict=True))
    norms = tuple(sqrt(sum(value * value for value in column)) for column in columns)
    normalized = [
        tuple(value / norm if norm else 0.0 for value, norm in zip(row, norms, strict=True))
        for row in (member.evaluation.objectives for member in front)
    ]
    weighted = [tuple(value * weight for value, weight in zip(row, weights, strict=True)) for row in normalized]
    ideal = tuple(min(row[index] for row in weighted) for index in range(3))
    anti_ideal = tuple(max(row[index] for row in weighted) for index in range(3))
    ranked: list[tuple[float, Individual]] = []
    for row, member in zip(weighted, front, strict=True):
        distance_ideal = sqrt(sum((value - ideal[index]) ** 2 for index, value in enumerate(row)))
        distance_anti = sqrt(sum((value - anti_ideal[index]) ** 2 for index, value in enumerate(row)))
        score = distance_anti / (distance_ideal + distance_anti) if distance_ideal + distance_anti else 1.0
        ranked.append((score, member))
    return sorted(ranked, key=lambda pair: (-pair[0], objective_sort_key(pair[1])))


def choose_representatives(front: tuple[Individual, ...]) -> dict[str, Individual]:
    if not front:
        return {}
    balanced = topsis_rank(front)[0][1]
    return {
        "lowest_cost": min(front, key=lambda member: (member.evaluation.total_cost, objective_sort_key(member))),
        "lowest_emissions": min(
            front,
            key=lambda member: (member.evaluation.total_emissions, objective_sort_key(member)),
        ),
        "highest_condition": max(
            front,
            key=lambda member: (
                member.evaluation.final_mean_condition,
                tuple(-value for value in member.evaluation.objectives[:2]),
            ),
        ),
        "balanced_topsis": balanced,
    }


def write_front_csv(path: Path, front: tuple[Individual, ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "rank",
                "total_cost",
                "total_emissions",
                "final_mean_condition",
                "min_condition_seen",
                "feasible",
                "violation",
            ]
        )
        for rank, individual in enumerate(front, start=1):
            evaluation = individual.evaluation
            writer.writerow(
                [
                    rank,
                    f"{evaluation.total_cost:.6f}",
                    f"{evaluation.total_emissions:.6f}",
                    f"{evaluation.final_mean_condition:.6f}",
                    f"{evaluation.min_condition_seen:.6f}",
                    str(evaluation.feasible).lower(),
                    f"{evaluation.violation:.6f}",
                ]
            )


def markdown_report(payload: dict[str, Any]) -> str:
    front = payload["front"]
    lines = [
        "# ParetoPilot benchmark",
        "",
        "A deterministic NSGA-II reference over a synthetic maintenance-planning scenario.",
        "",
        f"- population / generations: `{payload['populationSize']} / {payload['generations']}`",
        f"- final population: `{payload['finalPopulationSize']}` (`{payload['feasibleFinalPopulation']}` feasible)",
        f"- Pareto front: `{payload['paretoFrontSize']}` unique plans",
        f"- seed: `{payload['seed']}`",
        "",
        "| rank | cost | emissions | final mean | min seen | feasible |",
        "|---:|---:|---:|---:|---:|:---:|",
    ]
    for rank, member in enumerate(front, start=1):
        lines.append(
            f"| {rank} | {member['cost']:.2f} | {member['emissions']:.2f} | "
            f"{member['finalMeanCondition']:.2f} | {member['minConditionSeen']:.2f} | "
            f"{str(member['feasible']).lower()} |"
        )
    lines.extend(["", "## Representative choices", ""])
    for label, member in payload["representatives"].items():
        score = payload.get("representativeScores", {}).get(label)
        score_text = f", TOPSIS `{score:.3f}`" if score is not None else ""
        lines.append(
            f"- **{label}** — cost `{member['cost']:.2f}`, emissions `{member['emissions']:.2f}`, "
            f"final mean `{member['finalMeanCondition']:.2f}`{score_text}"
        )
    lines.extend(
        [
            "",
            "The fixture, plans, and metrics are synthetic. Objective values are a reproducibility "
            "contract, not a claim about a real road network, official maintenance policy, or production savings.",
            "",
        ]
    )
    return "\n".join(lines)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
