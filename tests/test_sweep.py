import hashlib
import json

import pytest

from paretopilot.optimizer import NSGA2Config
from paretopilot.sweep import parse_seed_list, seed_sweep_payload, write_sweep


def test_parse_seed_list_preserves_order_and_rejects_duplicates() -> None:
    assert parse_seed_list("17, 29,41") == (17, 29, 41)
    with pytest.raises(ValueError, match="unique"):
        parse_seed_list("17,17")
    with pytest.raises(ValueError, match="comma-separated"):
        parse_seed_list("17,not-a-seed")


def test_seed_sweep_is_deterministic_and_reports_representatives() -> None:
    config = NSGA2Config(population_size=16, generations=4, seed=99)
    first = seed_sweep_payload((17, 29), config)
    second = seed_sweep_payload((17, 29), config)
    assert first == second
    assert first["seeds"] == [17, 29]
    assert len(first["runs"]) == 2
    assert first["aggregate"]["balancedCost"]["pstdev"] >= 0


def test_seed_sweep_rejects_empty_or_duplicate_seeds() -> None:
    with pytest.raises(ValueError, match="at least one"):
        seed_sweep_payload(())
    with pytest.raises(ValueError, match="unique"):
        seed_sweep_payload((1, 1))


def test_write_sweep_emits_stable_manifest(tmp_path) -> None:
    config = NSGA2Config(population_size=12, generations=2, seed=17)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_sweep(first_dir, (17, 29), config)
    write_sweep(second_dir, (17, 29), config)
    expected = {"sweep.json", "sweep.csv", "report.md", "manifest.json"}
    assert {path.name for path in first_dir.iterdir()} == expected
    for name in expected:
        assert (first_dir / name).read_bytes() == (second_dir / name).read_bytes()
    manifest = json.loads((first_dir / "manifest.json").read_text())
    for entry in manifest["files"]:
        content = (first_dir / entry["name"]).read_bytes()
        assert entry["bytes"] == len(content)
        assert entry["sha256"] == hashlib.sha256(content).hexdigest()

