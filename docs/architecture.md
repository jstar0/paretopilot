# Architecture

```text
fixture.py -> Scenario/Plan -> evaluate_plan -> Evaluation
                                      |
                             constraint-domination
                                      |
                               optimizer.py (NSGA-II)
                                      |
                          report.py -> front + TOPSIS
                                      |
                         sweep.py -> multi-seed stability
```

`model.py` is the domain boundary. It validates immutable scenario inputs and
plan matrices, performs the year-by-year state transition, and exposes objective
and violation diagnostics. `optimizer.py` contains no file I/O: it only creates,
mutates, ranks, and selects plans. `report.py` turns an `OptimizationResult`
into stable artifacts and transparent representative choices; TOPSIS never
replaces the raw Pareto front. `sweep.py` repeats the same run across explicit
seeds and emits aggregate stability metrics. `cli.py` is a thin argument parser
around those layers.

The implementation intentionally avoids a framework and external data service.
That keeps a clean checkout auditable and makes a failed constraint, a changed
seed, or a different Pareto point easy to explain in an interview.
