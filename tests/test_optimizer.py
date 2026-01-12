import random

import pytest

from paretopilot.fixture import create_fixture
from paretopilot.model import Evaluation, Plan
from paretopilot.optimizer import (
    NSGA2Config,
    assign_crowding,
    crossover,
    fast_non_dominated_sort,
    make_individual,
    mutate,
    nsga2,
    random_plan,
    seed_plans,
)


def test_config_rejects_odd_or_invalid_population() -> None:
    with pytest.raises(ValueError, match="even"):
        NSGA2Config(population_size=5)
    with pytest.raises(ValueError, match="positive"):
        NSGA2Config(generations=0)


def test_random_plan_is_in_bounds() -> None:
    scenario = create_fixture()
    plan = random_plan(random.Random(17), scenario)
    assert plan.gene_count() == 72
    assert all(0 <= action < len(scenario.treatments) for row in plan.actions for action in row)


def test_seed_plans_include_baseline_and_heuristic() -> None:
    scenario = create_fixture()
    plans = seed_plans(scenario, random.Random(17))
    assert len(plans) >= 8
    assert plans[0] == Plan.no_action(scenario)
    assert len({plan.actions for plan in plans}) == len(plans)


def test_sort_places_non_dominated_members_in_first_front() -> None:
    scenario = create_fixture()
    plans = seed_plans(scenario, random.Random(17))[:4]
    individuals = [make_individual(plan, scenario) for plan in plans]
    fronts = fast_non_dominated_sort(individuals)
    assert fronts
    assert all(member.rank == 0 for member in fronts[0])


def test_crowding_marks_front_boundaries() -> None:
    scenario = create_fixture()
    plans = seed_plans(scenario, random.Random(17))[:5]
    individuals = [make_individual(plan, scenario) for plan in plans]
    assign_crowding(individuals)
    assert any(member.crowding == float("inf") for member in individuals)


def test_crossover_and_mutation_preserve_plan_shape() -> None:
    scenario = create_fixture()
    rng = random.Random(29)
    first, second = seed_plans(scenario, rng)[:2]
    child_a, child_b = crossover(first, second, scenario, rng, 1.0)
    assert child_a.gene_count() == first.gene_count()
    assert child_b.gene_count() == second.gene_count()
    changed = mutate(child_a, scenario, rng, 1.0)
    assert changed.gene_count() == child_a.gene_count()


def test_nsga2_is_deterministic_for_a_fixed_seed() -> None:
    config = NSGA2Config(population_size=24, generations=8, seed=17)
    first = nsga2(create_fixture(), config)
    second = nsga2(create_fixture(), config)
    assert [(member.evaluation.objectives, member.plan.actions) for member in first.pareto_front] == [
        (member.evaluation.objectives, member.plan.actions) for member in second.pareto_front
    ]
    assert first.generation_front_sizes == second.generation_front_sizes


def test_nsga2_front_is_unique_and_contains_feasible_plans() -> None:
    result = nsga2(create_fixture(), NSGA2Config(population_size=32, generations=10, seed=23))
    assert result.pareto_front
    assert len({member.plan.actions for member in result.pareto_front}) == len(result.pareto_front)
    assert all(member.evaluation.feasible for member in result.pareto_front)


def test_different_seed_can_explore_a_different_population() -> None:
    left = nsga2(create_fixture(), NSGA2Config(population_size=20, generations=5, seed=1))
    right = nsga2(create_fixture(), NSGA2Config(population_size=20, generations=5, seed=2))
    assert [member.plan.actions for member in left.final_population] != [
        member.plan.actions for member in right.final_population
    ]


def test_individual_factory_keeps_evaluation_bound_to_plan() -> None:
    scenario = create_fixture()
    individual = make_individual(Plan.no_action(scenario), scenario)
    assert isinstance(individual.evaluation, Evaluation)
    assert individual.plan == Plan.no_action(scenario)
