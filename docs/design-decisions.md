# ParetoPilot design decisions and limits

ParetoPilot is a synthetic constrained multi-objective optimization benchmark, not a real allocation or policy decision system.

## Decisions embodied in the implementation

1. **Constraint violation precedes objective dominance.** Infeasible plans cannot win by having a favorable objective. Rejected alternative: rank all candidates by weighted score first.
2. **Annual state transitions are explicit.** Costs, condition and budgets evolve across the scenario horizon. Rejected alternative: optimize each year independently.
3. **NSGA-II produces a candidate frontier; TOPSIS selects a transparent representative.** Rejected alternative: present a single optimizer output as the only valid answer.
4. **Feasible representatives are classified by stated policy profiles.** Rejected alternative: hide tradeoffs behind one global utility weight.
5. **Seed sweeps and hash-bound artifacts expose fixture stability.** Rejected alternative: report one favorable stochastic run.

## Known weaknesses

1. **Assets, budgets, constraints, and objectives are synthetic.**
2. **NSGA-II and TOPSIS settings are benchmark choices, not validated policy preferences.**
3. **Objectives omit stakeholders, uncertainty, equity, regulation, and real operational costs.**
4. **The representative selection is sensitive to normalization and TOPSIS weights.**
5. **No exact optimum, lower bound, or published-instance comparison is provided.**
6. **Seed stability on one fixture does not establish robustness under distribution shift.**
7. **There is no authorization, workflow, persistence, or production decision service.**

## Interview-safe description

Describe ParetoPilot as a reproducible synthetic constrained multi-objective lab with explicit feasibility and representative-selection tradeoffs.
