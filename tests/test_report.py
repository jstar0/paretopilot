import hashlib
import json

from paretopilot.fixture import create_fixture
from paretopilot.optimizer import NSGA2Config, nsga2
from paretopilot.report import (
    benchmark_payload,
    build_baselines,
    topsis_rank,
    write_benchmark,
)


def test_payload_contains_front_baselines_and_generation_trace() -> None:
    result = nsga2(create_fixture(), NSGA2Config(population_size=20, generations=5, seed=17))
    payload = benchmark_payload(result)
    assert payload["benchmarkVersion"] == 1
    assert payload["paretoFrontSize"] == len(result.pareto_front)
    assert set(payload["baselines"]) == {"no_action", "light_cycle", "performance_cycle"}
    assert set(payload["representatives"]) == {
        "lowest_cost",
        "lowest_emissions",
        "highest_condition",
        "balanced_topsis",
    }
    assert set(payload["representativeScores"]) == set(payload["representatives"])
    assert len(payload["generationFrontSizes"]) == 5


def test_write_benchmark_emits_all_public_artifacts(tmp_path) -> None:
    write_benchmark(tmp_path, NSGA2Config(population_size=20, generations=5, seed=17))
    expected = {"benchmark.json", "pareto_front.csv", "report.md", "manifest.json"}
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert "| rank | cost | emissions |" in (tmp_path / "report.md").read_text()


def test_write_benchmark_is_byte_stable(tmp_path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    config = NSGA2Config(population_size=20, generations=5, seed=17)
    write_benchmark(first_dir, config)
    write_benchmark(second_dir, config)
    for name in ("benchmark.json", "pareto_front.csv", "report.md", "manifest.json"):
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()


def test_manifest_hashes_match_artifacts(tmp_path) -> None:
    write_benchmark(tmp_path, NSGA2Config(population_size=20, generations=5, seed=17))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for entry in manifest["files"]:
        content = (tmp_path / entry["name"]).read_bytes()
        assert entry["bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()


def test_baselines_match_scenario_dimensions() -> None:
    scenario = create_fixture()
    for plan in build_baselines(scenario).values():
        assert plan.gene_count() == len(scenario.segments) * scenario.years


def test_topsis_ranks_feasible_members_and_is_deterministic() -> None:
    result = nsga2(create_fixture(), NSGA2Config(population_size=20, generations=5, seed=17))
    first = topsis_rank(result.pareto_front)
    second = topsis_rank(result.pareto_front)
    assert first
    assert [(score, member.plan.actions) for score, member in first] == [
        (score, member.plan.actions) for score, member in second
    ]
    assert all(0.0 <= score <= 1.0 for score, _ in first)
    assert all(member.evaluation.feasible for _, member in first)


def test_topsis_rejects_invalid_weights() -> None:
    result = nsga2(create_fixture(), NSGA2Config(population_size=12, generations=2, seed=17))
    import pytest

    with pytest.raises(ValueError, match="weights"):
        topsis_rank(result.pareto_front, (float("nan"), 0.3, 0.7))
    with pytest.raises(ValueError, match="weights"):
        topsis_rank(result.pareto_front, (0.0, 0.0, 0.0))
