"""Command-line entry point for reproducible ParetoPilot artifacts."""

from __future__ import annotations

import argparse
import json

from .optimizer import NSGA2Config
from .report import write_benchmark
from .sweep import parse_seed_list, write_sweep


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="paretopilot", description="Deterministic synthetic multi-objective planning lab"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    benchmark = subparsers.add_parser("benchmark", help="run NSGA-II and write reproducible artifacts")
    benchmark.add_argument("--output-dir", default="artifacts/benchmark")
    benchmark.add_argument("--seed", type=int, default=17)
    benchmark.add_argument("--population", type=int, default=64)
    benchmark.add_argument("--generations", type=int, default=35)
    sweep = subparsers.add_parser("sweep", help="run a deterministic multi-seed stability sweep")
    sweep.add_argument("--output-dir", default="artifacts/sweep")
    sweep.add_argument("--seeds", default="17,29,41")
    sweep.add_argument("--population", type=int, default=64)
    sweep.add_argument("--generations", type=int, default=35)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "benchmark":
        payload = write_benchmark(
            args.output_dir,
            NSGA2Config(seed=args.seed, population_size=args.population, generations=args.generations),
        )
        summary = {
            "output_dir": args.output_dir,
            "pareto_front_size": payload["paretoFrontSize"],
            "seed": payload["seed"],
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    if args.command == "sweep":
        payload = write_sweep(
            args.output_dir,
            parse_seed_list(args.seeds),
            NSGA2Config(population_size=args.population, generations=args.generations),
        )
        summary = {
            "output_dir": args.output_dir,
            "runs": len(payload["runs"]),
            "seeds": payload["seeds"],
        }
        print(json.dumps(summary, sort_keys=True))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
