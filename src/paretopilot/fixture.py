"""A fixed, synthetic scenario for examples and regression benchmarks."""

from __future__ import annotations

from .model import Scenario, Segment, Treatment


def create_fixture() -> Scenario:
    segments = tuple(
        Segment(
            id=f"seg-{index:02d}",
            initial_condition=88.0 + ((index * 7) % 11),
            annual_decay=1.25 + ((index * 3) % 5) * 0.18,
        )
        for index in range(12)
    )
    treatments = (
        Treatment("none", "No action", 0.0, 0.0, 0.0, 0.0),
        Treatment("seal", "Surface seal", 34.0, 8.0, 4.0, 0.18),
        Treatment("overlay", "Thin overlay", 67.0, 17.0, 8.0, 0.34),
        Treatment("recycle", "Deep recycle", 104.0, 28.0, 13.0, 0.5),
    )
    return Scenario(
        segments=segments,
        treatments=treatments,
        years=6,
        annual_budget=460.0,
        min_condition=80.0,
        min_final_mean=84.0,
    )
