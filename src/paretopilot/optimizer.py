"""A compact, deterministic NSGA-II implementation."""

from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass

from .model import (
    Evaluation,
    Plan,
    Scenario,
    constraint_dominates,
    evaluate_plan,
)


@dataclass(frozen=True, slots=True)
class NSGA2Config:
    population_size: int = 64
    generations: int = 35
    crossover_rate: float = 0.9
    mutation_rate: float = 0.04
    seed: int = 17

    def __post_init__(self) -> None:
        if self.population_size < 4 or self.population_size % 2:
            raise ValueError("population_size must be an even number >= 4")
        if self.generations < 1:
            raise ValueError("generations must be positive")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("crossover and mutation rates must be in [0, 1]")


@dataclass(slots=True)
class Individual:
    plan: Plan
    evaluation: Evaluation
    rank: int = 0
    crowding: float = 0.0


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    scenario: Scenario
    config: NSGA2Config
    pareto_front: tuple[Individual, ...]
    final_population: tuple[Individual, ...]
    generation_front_sizes: tuple[int, ...]


def make_individual(plan: Plan, scenario: Scenario) -> Individual:
    return Individual(plan=plan, evaluation=evaluate_plan(plan, scenario))


def fast_non_dominated_sort(population: list[Individual]) -> list[list[Individual]]:
    index_by_identity = {id(member): index for index, member in enumerate(population)}
    domination_count = [0] * len(population)
    dominated: list[list[int]] = [[] for _ in population]
    fronts: list[list[Individual]] = [[]]
    for index, left in enumerate(population):
        for other_index, right in enumerate(population):
            if index == other_index:
                continue
            if constraint_dominates(left.evaluation, right.evaluation):
                dominated[index].append(other_index)
            elif constraint_dominates(right.evaluation, left.evaluation):
                domination_count[index] += 1
        if domination_count[index] == 0:
            left.rank = 0
            fronts[0].append(left)
    cursor = 0
    while cursor < len(fronts) and fronts[cursor]:
        next_front: list[Individual] = []
        for member in fronts[cursor]:
            member_index = index_by_identity[id(member)]
            for dominated_index in dominated[member_index]:
                domination_count[dominated_index] -= 1
                if domination_count[dominated_index] == 0:
                    population[dominated_index].rank = cursor + 1
                    next_front.append(population[dominated_index])
        cursor += 1
        if next_front:
            fronts.append(next_front)
    return fronts


def assign_crowding(front: list[Individual]) -> None:
    if not front:
        return
    for member in front:
        member.crowding = 0.0
    if len(front) <= 2:
        for member in front:
            member.crowding = float("inf")
        return
    objective_count = len(front[0].evaluation.objectives)
    for objective_index in range(objective_count):
        ordered = sorted(front, key=lambda item: item.evaluation.objectives[objective_index])
        ordered[0].crowding = float("inf")
        ordered[-1].crowding = float("inf")
        minimum = ordered[0].evaluation.objectives[objective_index]
        maximum = ordered[-1].evaluation.objectives[objective_index]
        spread = maximum - minimum
        if spread <= 1e-12:
            continue
        for index in range(1, len(ordered) - 1):
            if ordered[index].crowding == float("inf"):
                continue
            previous_value = ordered[index - 1].evaluation.objectives[objective_index]
            next_value = ordered[index + 1].evaluation.objectives[objective_index]
            ordered[index].crowding += (next_value - previous_value) / spread


def rank_population(population: list[Individual]) -> list[list[Individual]]:
    fronts = fast_non_dominated_sort(population)
    for front in fronts:
        assign_crowding(front)
    return fronts


def select_survivors(candidates: Iterable[Individual], size: int) -> list[Individual]:
    pool = list(candidates)
    fronts = rank_population(pool)
    survivors: list[Individual] = []
    for front in fronts:
        if len(survivors) + len(front) <= size:
            survivors.extend(front)
            continue
        assign_crowding(front)
        front.sort(key=lambda member: (-member.crowding, member.plan.actions))
        survivors.extend(front[: size - len(survivors)])
        break
    return survivors


def random_plan(rng: random.Random, scenario: Scenario) -> Plan:
    rows = []
    for _ in scenario.segments:
        row = []
        for _ in range(scenario.years):
            # Most years do nothing; the heavier interventions remain visible
            # often enough to create cost/performance trade-offs.
            draw = rng.random()
            action = 0 if draw < 0.56 else 1 if draw < 0.78 else 2 if draw < 0.95 else 3
            row.append(action)
        rows.append(row)
    return Plan.from_rows(rows, scenario)


def seed_plans(scenario: Scenario, rng: random.Random) -> list[Plan]:
    plans = [Plan.no_action(scenario)]
    for action in range(1, len(scenario.treatments)):
        rows = [[action if year % 3 == 0 else 0 for year in range(scenario.years)] for _ in scenario.segments]
        plans.append(Plan.from_rows(rows, scenario))
    # A severity-aware scaffold spends heavier treatments on the weakest
    # segments and light treatments elsewhere, producing a useful anchor.
    severity = sorted(range(len(scenario.segments)), key=lambda i: scenario.segments[i].initial_condition)
    rows = [[0 for _ in range(scenario.years)] for _ in scenario.segments]
    for rank, segment_index in enumerate(severity):
        action = 3 if rank < len(severity) // 3 else 2 if rank < (2 * len(severity)) // 3 else 1
        for year in range(0, scenario.years, 3):
            rows[segment_index][year] = action
    plans.append(Plan.from_rows(rows, scenario))
    while len(plans) < 8:
        plans.append(random_plan(rng, scenario))
    return plans


DEFAULT_CONFIG = NSGA2Config()


def nsga2(scenario: Scenario, config: NSGA2Config | None = None) -> OptimizationResult:
    config = DEFAULT_CONFIG if config is None else config
    rng = random.Random(config.seed)
    plans = seed_plans(scenario, rng)
    while len(plans) < config.population_size:
        plans.append(random_plan(rng, scenario))
    population = select_survivors((make_individual(plan, scenario) for plan in plans), config.population_size)
    generation_front_sizes: list[int] = []

    for _ in range(config.generations):
        fronts = rank_population(population)
        generation_front_sizes.append(len(fronts[0]) if fronts else 0)
        offspring: list[Individual] = []
        while len(offspring) < config.population_size:
            first = tournament(rng, population)
            second = tournament(rng, population)
            child_a, child_b = crossover(first.plan, second.plan, scenario, rng, config.crossover_rate)
            offspring.append(make_individual(mutate(child_a, scenario, rng, config.mutation_rate), scenario))
            if len(offspring) < config.population_size:
                offspring.append(make_individual(mutate(child_b, scenario, rng, config.mutation_rate), scenario))
        population = select_survivors([*population, *offspring], config.population_size)

    fronts = rank_population(population)
    front = tuple(sorted(unique_individuals(fronts[0] if fronts else []), key=objective_sort_key))
    return OptimizationResult(
        scenario=scenario,
        config=config,
        pareto_front=front,
        final_population=tuple(population),
        generation_front_sizes=tuple(generation_front_sizes),
    )


def tournament(rng: random.Random, population: list[Individual]) -> Individual:
    left, right = rng.sample(population, 2)
    if left.rank != right.rank:
        return left if left.rank < right.rank else right
    if left.crowding != right.crowding:
        return left if left.crowding > right.crowding else right
    return left if left.plan.actions <= right.plan.actions else right


def crossover(
    first: Plan,
    second: Plan,
    scenario: Scenario,
    rng: random.Random,
    probability: float,
) -> tuple[Plan, Plan]:
    if rng.random() >= probability:
        return first, second
    first_genes = [action for row in first.actions for action in row]
    second_genes = [action for row in second.actions for action in row]
    point = rng.randrange(1, len(first_genes))
    child_genes = (first_genes[:point] + second_genes[point:], second_genes[:point] + first_genes[point:])
    return tuple(_plan_from_genes(genes, scenario) for genes in child_genes)  # type: ignore[return-value]


def mutate(plan: Plan, scenario: Scenario, rng: random.Random, probability: float) -> Plan:
    genes = [action for row in plan.actions for action in row]
    for index in range(len(genes)):
        if rng.random() < probability:
            genes[index] = rng.randrange(len(scenario.treatments))
    return _plan_from_genes(genes, scenario)


def _plan_from_genes(genes: list[int], scenario: Scenario) -> Plan:
    rows = []
    cursor = 0
    for _ in scenario.segments:
        rows.append(genes[cursor : cursor + scenario.years])
        cursor += scenario.years
    return Plan.from_rows(rows, scenario)


def unique_individuals(individuals: Iterable[Individual]) -> list[Individual]:
    seen: set[tuple[tuple[int, ...], ...]] = set()
    unique: list[Individual] = []
    for individual in individuals:
        if individual.plan.actions in seen:
            continue
        seen.add(individual.plan.actions)
        unique.append(individual)
    return unique


def objective_sort_key(individual: Individual) -> tuple[float, float, float, tuple[tuple[int, ...], ...]]:
    return (*individual.evaluation.objectives, individual.plan.actions)
