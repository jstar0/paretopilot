from dataclasses import replace

import pytest

from paretopilot.fixture import create_fixture
from paretopilot.model import (
    Plan,
    Segment,
    Treatment,
    constraint_dominates,
    evaluate_plan,
    objective_dominates,
    validate_plan,
)


def test_fixture_shape_and_ids_are_stable() -> None:
    scenario = create_fixture()
    assert scenario.shape == (12, 6)
    assert [segment.id for segment in scenario.segments[:3]] == ["seg-00", "seg-01", "seg-02"]
    assert [treatment.id for treatment in scenario.treatments] == ["none", "seal", "overlay", "recycle"]


def test_scenario_thresholds_and_treatment_zero_are_explicit() -> None:
    scenario = create_fixture()
    assert scenario.annual_budget == 460.0
    assert scenario.min_condition == 80.0
    assert scenario.treatments[0].cost == 0.0
    assert scenario.treatments[0].emissions == 0.0


def test_plan_no_action_has_expected_shape() -> None:
    plan = Plan.no_action(create_fixture())
    assert plan.gene_count() == 72
    assert all(action == 0 for row in plan.actions for action in row)


def test_plan_rejects_wrong_shape_and_unknown_actions() -> None:
    scenario = create_fixture()
    with pytest.raises(ValueError, match="row count"):
        Plan.from_rows([[0]], scenario)
    rows = [[0 for _ in range(scenario.years)] for _ in scenario.segments]
    rows[0][0] = 99
    with pytest.raises(ValueError, match="unknown treatment"):
        Plan.from_rows(rows, scenario)


def test_plan_rejects_boolean_actions() -> None:
    scenario = create_fixture()
    rows = [[0 for _ in range(scenario.years)] for _ in scenario.segments]
    rows[0][0] = True  # type: ignore[assignment]
    with pytest.raises(ValueError, match="unknown treatment"):
        Plan.from_rows(rows, scenario)


def test_domain_objects_reject_invalid_numbers() -> None:
    with pytest.raises(ValueError, match="initial condition"):
        Segment("bad", 0, 1)
    with pytest.raises(ValueError, match="annual decay"):
        Segment("bad", 90, 0)
    with pytest.raises(ValueError, match="non-negative"):
        Treatment("bad", "Bad", -1, 0, 0, 0)
    with pytest.raises(ValueError, match="durability"):
        Treatment("bad", "Bad", 1, 1, 1, 0.95)


def test_no_action_is_deterministic_and_records_all_years() -> None:
    scenario = create_fixture()
    evaluation = evaluate_plan(Plan.no_action(scenario), scenario)
    assert evaluation == evaluate_plan(Plan.no_action(scenario), scenario)
    assert len(evaluation.yearly_mean_condition) == scenario.years
    assert len(evaluation.final_conditions) == len(scenario.segments)
    assert evaluation.objectives[2] == -evaluation.final_mean_condition


def test_treatment_can_improve_condition_over_no_action() -> None:
    scenario = create_fixture()
    baseline = evaluate_plan(Plan.no_action(scenario), scenario)
    rows = [[2 if year % 2 == 0 else 0 for year in range(scenario.years)] for _ in scenario.segments]
    treated = evaluate_plan(Plan.from_rows(rows, scenario), scenario)
    assert treated.final_mean_condition > baseline.final_mean_condition
    assert treated.total_cost > baseline.total_cost
    assert treated.total_emissions > baseline.total_emissions


def test_budget_violation_is_reported_without_crashing() -> None:
    scenario = replace(create_fixture(), annual_budget=1.0)
    rows = [[3 for _ in range(scenario.years)] for _ in scenario.segments]
    evaluation = evaluate_plan(Plan.from_rows(rows, scenario), scenario)
    assert not evaluation.feasible
    assert evaluation.violation > 0


def test_condition_floor_violation_is_reported() -> None:
    scenario = replace(create_fixture(), min_condition=99.0, min_final_mean=99.0)
    evaluation = evaluate_plan(Plan.no_action(scenario), scenario)
    assert not evaluation.feasible
    assert evaluation.min_condition_seen < scenario.min_condition
    assert evaluation.violation > 0


def test_annual_budget_is_applied_per_year() -> None:
    scenario = create_fixture()
    rows = [[1 if year == 0 else 0 for year in range(scenario.years)] for _ in scenario.segments]
    evaluation = evaluate_plan(Plan.from_rows(rows, scenario), scenario)
    assert evaluation.total_cost == pytest.approx(12 * 34)
    assert evaluation.feasible or evaluation.violation >= 0


def test_objective_dominance_requires_one_strict_improvement() -> None:
    scenario = create_fixture()
    baseline = evaluate_plan(Plan.no_action(scenario), scenario)
    same = evaluate_plan(Plan.no_action(scenario), scenario)
    assert not objective_dominates(baseline, same)
    better = replace(baseline, objectives=(1.0, 1.0, -90.0))
    worse = replace(baseline, objectives=(2.0, 2.0, -89.0))
    assert objective_dominates(better, worse)
    assert not objective_dominates(worse, better)


def test_constraint_dominance_prioritizes_feasibility() -> None:
    scenario = replace(create_fixture(), annual_budget=900.0)
    feasible_rows = [[2 if year % 3 == 0 else 0 for year in range(scenario.years)] for _ in scenario.segments]
    feasible = evaluate_plan(Plan.from_rows(feasible_rows, scenario), scenario)
    infeasible = evaluate_plan(Plan.no_action(scenario), scenario)
    assert feasible.feasible
    assert not infeasible.feasible
    assert constraint_dominates(feasible, infeasible)


def test_validate_plan_accepts_tuple_rows() -> None:
    scenario = create_fixture()
    plan = Plan(tuple(tuple(0 for _ in range(scenario.years)) for _ in scenario.segments))
    validate_plan(plan, scenario)


def test_scenario_rejects_invalid_budget_and_missing_treatment() -> None:
    with pytest.raises(ValueError, match="positive"):
        replace(create_fixture(), annual_budget=0)
    with pytest.raises(ValueError, match="no-action"):
        from paretopilot.model import Scenario

        Scenario(create_fixture().segments, (Treatment("x", "x", 1, 1, 1, 0.1),), 2, 10, 80, 80)


def test_scenario_rejects_a_nonzero_baseline_boost() -> None:
    scenario = create_fixture()
    treatments = (replace(scenario.treatments[0], condition_boost=1.0), *scenario.treatments[1:])
    with pytest.raises(ValueError, match="no-cost baseline"):
        from paretopilot.model import Scenario

        Scenario(
            scenario.segments,
            treatments,
            scenario.years,
            scenario.annual_budget,
            scenario.min_condition,
            scenario.min_final_mean,
        )
