"""Typed domain model and deterministic evaluator for ParetoPilot.

The public model is deliberately small: a plan is a matrix of treatment
indices, while :func:`evaluate_plan` is the only place that advances road
condition state.  Keeping that boundary explicit makes the optimizer and the
reporting layer independently testable.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

Objective: TypeAlias = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Segment:
    """A synthetic asset with an initial condition and natural decay rate."""

    id: str
    initial_condition: float
    annual_decay: float

    def __post_init__(self) -> None:
        if not self.id or not self.id.replace("-", "").isalnum():
            raise ValueError("segment id must be non-empty and identifier-like")
        if not isfinite(self.initial_condition) or not 0 < self.initial_condition <= 100:
            raise ValueError("initial condition must be in (0, 100]")
        if not isfinite(self.annual_decay) or self.annual_decay <= 0:
            raise ValueError("annual decay must be positive")


@dataclass(frozen=True, slots=True)
class Treatment:
    """One intervention option; index zero is conventionally no action."""

    id: str
    name: str
    cost: float
    emissions: float
    condition_boost: float
    durability: float

    def __post_init__(self) -> None:
        if not self.id or not self.name:
            raise ValueError("treatment id and name are required")
        for value in (self.cost, self.emissions, self.condition_boost, self.durability):
            if not isfinite(value) or value < 0:
                raise ValueError("treatment numeric fields must be finite and non-negative")
        if self.durability > 0.9:
            raise ValueError("durability must be at most 0.9")


@dataclass(frozen=True, slots=True)
class Scenario:
    """Optimization constraints and immutable synthetic input data."""

    segments: tuple[Segment, ...]
    treatments: tuple[Treatment, ...]
    years: int
    annual_budget: float
    min_condition: float
    min_final_mean: float

    def __post_init__(self) -> None:
        if not self.segments:
            raise ValueError("scenario needs at least one segment")
        if len(self.treatments) < 2:
            raise ValueError("scenario needs no-action plus one treatment")
        if (
            self.treatments[0].cost != 0
            or self.treatments[0].emissions != 0
            or self.treatments[0].condition_boost != 0
            or self.treatments[0].durability != 0
        ):
            raise ValueError("treatment zero must be the no-cost baseline")
        if self.years < 1 or self.years > 30:
            raise ValueError("years must be between 1 and 30")
        if not isfinite(self.annual_budget) or self.annual_budget <= 0:
            raise ValueError("annual budget must be positive")
        if not 0 < self.min_condition <= 100 or not 0 < self.min_final_mean <= 100:
            raise ValueError("condition thresholds must be in (0, 100]")

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.segments), self.years)


@dataclass(frozen=True, slots=True)
class Plan:
    """A segment-by-year matrix, stored as immutable tuples."""

    actions: tuple[tuple[int, ...], ...]

    @classmethod
    def no_action(cls, scenario: Scenario) -> Plan:
        return cls(tuple(tuple(0 for _ in range(scenario.years)) for _ in scenario.segments))

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[int]], scenario: Scenario) -> Plan:
        plan = cls(tuple(tuple(row) for row in rows))
        validate_plan(plan, scenario)
        return plan

    def gene_count(self) -> int:
        return sum(len(row) for row in self.actions)

    def with_gene(self, segment_index: int, year_index: int, action: int, scenario: Scenario) -> Plan:
        rows = [list(row) for row in self.actions]
        rows[segment_index][year_index] = action
        return Plan.from_rows(rows, scenario)


@dataclass(frozen=True, slots=True)
class Evaluation:
    """Objective values plus enough diagnostics to explain feasibility."""

    objectives: Objective
    total_cost: float
    total_emissions: float
    final_mean_condition: float
    min_condition_seen: float
    feasible: bool
    violation: float
    yearly_mean_condition: tuple[float, ...]
    final_conditions: tuple[float, ...]


def validate_plan(plan: Plan, scenario: Scenario) -> None:
    if len(plan.actions) != len(scenario.segments):
        raise ValueError("plan row count does not match scenario")
    for row in plan.actions:
        if len(row) != scenario.years:
            raise ValueError("plan column count does not match scenario years")
        for action in row:
            if not isinstance(action, int) or isinstance(action, bool) or not 0 <= action < len(scenario.treatments):
                raise ValueError("plan contains an unknown treatment index")


def evaluate_plan(plan: Plan, scenario: Scenario) -> Evaluation:
    """Simulate a plan and return three minimization objectives.

    Objectives are total cost, total emissions, and the negative final mean
    condition (so that all three can use the same Pareto minimization rule).
    Treatment boosts happen at the start of a year; natural decay then applies
    with a durability-adjusted factor.  A plan is infeasible when it exceeds an
    annual budget, dips below the condition floor, or misses the final mean.
    """

    validate_plan(plan, scenario)
    conditions = [segment.initial_condition for segment in scenario.segments]
    years_since = [0 for _ in scenario.segments]
    yearly_mean: list[float] = []
    total_cost = 0.0
    total_emissions = 0.0
    violation = 0.0
    min_seen = min(conditions)

    for year in range(scenario.years):
        annual_cost = 0.0
        for index, segment in enumerate(scenario.segments):
            action = plan.actions[index][year]
            treatment = scenario.treatments[action]
            annual_cost += treatment.cost
            total_cost += treatment.cost
            total_emissions += treatment.emissions
            if action == 0:
                years_since[index] += 1
                decay_factor = 1.0 + 0.06 * min(years_since[index], 5)
            else:
                years_since[index] = 0
                decay_factor = max(0.55, 1.0 - treatment.durability)
            conditions[index] = min(100.0, conditions[index] + treatment.condition_boost)
            conditions[index] = max(0.0, conditions[index] - segment.annual_decay * decay_factor)
            min_seen = min(min_seen, conditions[index])
            if conditions[index] < scenario.min_condition:
                violation += scenario.min_condition - conditions[index]
        if annual_cost > scenario.annual_budget:
            violation += annual_cost - scenario.annual_budget
        yearly_mean.append(sum(conditions) / len(conditions))

    final_mean = sum(conditions) / len(conditions)
    if final_mean < scenario.min_final_mean:
        violation += scenario.min_final_mean - final_mean
    objectives: Objective = (total_cost, total_emissions, -final_mean)
    return Evaluation(
        objectives=objectives,
        total_cost=total_cost,
        total_emissions=total_emissions,
        final_mean_condition=final_mean,
        min_condition_seen=min_seen,
        feasible=violation <= 1e-9,
        violation=violation,
        yearly_mean_condition=tuple(yearly_mean),
        final_conditions=tuple(conditions),
    )


def objective_dominates(left: Evaluation, right: Evaluation) -> bool:
    """Return true when ``left`` dominates ``right`` under minimization."""

    return all(a <= b + 1e-12 for a, b in zip(left.objectives, right.objectives, strict=True)) and any(
        a < b - 1e-12 for a, b in zip(left.objectives, right.objectives, strict=True)
    )


def constraint_dominates(left: Evaluation, right: Evaluation) -> bool:
    """Constraint-domination used by NSGA-II."""

    if left.feasible != right.feasible:
        return left.feasible
    if not left.feasible and left.violation != right.violation:
        return left.violation < right.violation
    return objective_dominates(left, right)


def plan_to_dict(plan: Plan) -> dict[str, list[list[int]]]:
    return {"actions": [list(row) for row in plan.actions]}
