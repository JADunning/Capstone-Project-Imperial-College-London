# BBO Capstone Project — Presentation

Imperial College London · Machine Learning Capstone · Black-Box Optimization

---

This document presents an overview of the Black-Box Optimization (BBO) capstone: objectives, strategy, evolution, insights, decision-making, and next steps. For setup and technical details see the [README](README.md); for reproducibility see the [Model Card](MODEL_CARD.md) and [Datasheet](DATASHEET.md).

---

## 1. Overview of the BBO Approach

### What we are trying to achieve

Maximise the output of eight unknown, expensive black-box functions with very limited evaluations—roughly one query per function per week over 12 weeks (~96 queries total). We only observe input–output pairs. The challenge is to use those observations to find the best input configurations before the evaluation budget runs out. In plain terms: discover where each “hidden” function is highest by balancing learning about the space (exploration) with capitalising on what we already know (exploitation).

### Overall strategy and how it fits together

The approach is surrogate-based optimisation: we fit a model to the observed data and use it to decide where to query next. The strategy is function-specific:

- Function 1 (2D, sparse/near-binary): Non-linear logistic regression learns a decision boundary; we query near it (e.g. probability of “good” ~0.5 or 0.9).
- Functions 2, 4, 5, 6 (2D–5D): GP surrogate with UCB or Expected Improvement; exploitative UCB (β = 0), switch to EI when progress stalls.
- Function 3 (3D, noisy): Noise-aware BO (GP + WhiteKernel), local restarts near the current best.
- Functions 7 and 8 (6D, 8D): PyTorch MLP fitted to the data; next point by gradient-based optimisation from multiple restarts.

Over time the process moves from early exploration (weeks 1–3) through transition (weeks 4–6) to exploitation and refinement (weeks 7–12). Checking whether the latest submission improved and how far suggestions are from the incumbent guides where to focus each week.

---

## 2. How the Strategy Has Evolved

### Key changes since the early rounds

- From uniform exploration to function-specific methods: First weeks used the same exploration BO (GP + UCB, β = 2) for all functions. As data accumulated, the strategy split by function: logistic regression for Function 1, noise-aware BO for Function 3, and NN+gradient for Functions 7 and 8.

- From exploration to exploitation, then local trust-region BO, then EI when stuck: After building surrogates, the approach shifted to exploitation (UCB β = 0). When that drifted or plateaued (repeats created spurious GP peaks), local trust-region BO was introduced (duplicate aggregation, WhiteKernel, maximise GP mean only near the incumbent). When progress still stalled, Expected Improvement rebalanced exploration and exploitation.

- From ad hoc inspection to structured validation: Early analysis used simple 2D plots. t-SNE was introduced in week 2 with the first BO pipeline. In week 8 the results table was introduced: Source, Output value, Distance to maximum input, Summary (Improving/Not improving). For Function 1, a class-assignments table with Rank and Size bar was added in week 11.

### What influenced these changes

- Data trends: Flat or sparse behaviour (Function 1), noisy outputs (Function 3), and high dimensionality (7, 8) drove surrogate and acquisition choice.
- Feedback: Week-on-week results showed when a method was not improving and triggered switches (e.g. to EI).
- Model performance: Low surrogate MSE increased confidence in exploitation; high MSE or unstable suggestions prompted more exploration or a different model.
- Intuition: "One method per function" to reduce redundancy and match the problem guided the move to specialised pipelines.

### Principles that guide query decisions now

- Match the method to the function: Sparse/binary → logistic regression; noisy → noise-aware GP; high-dimensional → NN + gradient; smooth moderate-dimensional → GP + UCB/EI.
- Validate progress: Use tables to see which functions improved and which stalled; focus on stallers.
- Exploit when the surrogate is trustworthy, explore when it isn't; when exploitation stalls, switch to EI to rebalance.

---

## 3. Patterns, Data and Insights

### Meaningful trends

- Function 1: Most of the input space yields near-zero output; only a small region is “good.” The logistic boundary (with polynomial features) captures this; querying near it (0.5 or 0.9 contour) refines the best region.
- Function 3: Noisy, often negative outputs; noise-aware BO and occasional re-runs avoid locking onto spurious peaks.
- Functions 5, 7, 8: Once a high-value region was found, gains became incremental; strategy shifted to local refinement.
- Functions 2, 4, 6: Mixed behaviour; exploitative BO sometimes drifted. EI and progress validation helped refocus.

### Variables that influence results most

- Function identity and dimensionality; noise (Function 3); strategy–problem match; distance to incumbent (close = exploitation, far = exploration).

### How these observations shape understanding

The search is iterative and adaptive: early data reveal function type, which dictates the surrogate and acquisition. Variance in outcomes is largely explained by function type and strategy–problem fit. Focusing on dimensions the surrogate says matter most (e.g. logistic coefficients, GP length scales) keeps essential structure and reduces redundant or noisy dimensions when deciding where to query next.

---

## 4. Decision-Making and Iteration

### Balancing exploration and exploitation

Early rounds (1–3): emphasis on exploration. Middle and later rounds (4–10): shift to exploitation (UCB β = 0, local refinement). When progress stalls or suggestions drift far from the best, increase exploration via EI or re-check surrogate assumptions. Progress validation and distance-to-incumbent tables indicate over-exploiting vs over-exploring and inform the next week’s choice.

### Examples of strategic decisions

- Function 1 — switching from BO to logistic regression: BO suggested extreme points (e.g. corners) because the function is mostly flat. Decision: use logistic regression to learn the boundary and query near it. Result: more stable improvement. This worked because it matched the problem structure.

- Exploitative BO drifting: UCB β = 0 kept suggesting points that did not improve the best or were far from the incumbent. Decision: introduce EI and progress validation; use EI when the incumbent had not improved. Result: better balance. What didn’t work was continuing with pure exploitation without a formal progress check.

### Handling uncertainty

If a suggested point performs poorly or very differently from the surrogate’s prediction, we treat it as new data and refit; for Function 3 we allow for noise and sometimes re-run points. When results don’t match expectations we use validation and surrogate diagnostics; if the surrogate is unreliable or progress stalls, we change method or acquisition rather than repeating the same strategy.

---

## 5. Next Steps and Reflection

### Planned actions

- Final round: one more exploitation pass—current best method per function, refine around best-known points.
- If the project continued: tighten NN and BO settings; formalise progress validation (e.g. automatic “switch to EI if no improvement for k weeks”); consider meta-analysis across functions to prioritise tuning.

### Connection to the broader ML landscape

The project sits at the intersection of Bayesian optimisation, surrogate-based optimisation, and experimental design. The same ideas underpin hyperparameter tuning (e.g. Optuna, SMAC), materials discovery, and drug design. Method choice and exploration–exploitation balance are as important as the surrogate; documentation and validation support reproducibility and robust decision-making.

### Communicating to a non-technical audience

“We had a limited number of expensive ‘experiments’ on eight unknown systems. We used past results to build simple predictive models and chose each new experiment to learn more or to improve on our best so far. We tailored the approach to each system (e.g. different strategy for noisy or high-dimensional ones) and tracked whether we were actually improving. We improved over time by combining exploration and exploitation and adapting the method to what the data showed.”

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [README](README.md) | Project overview, setup, inputs/outputs, technical approach, weekly strategy docs |
| [MODEL_CARD.md](MODEL_CARD.md) | Optimisation strategy, intended use, performance, limitations |
| [DATASHEET.md](DATASHEET.md) | Dataset motivation, composition, collection, preprocessing |
| [Weekly strategy (week 1–11)](README.md#weekly-strategy-documents) | Week-by-week changes: method, visualisation, and why |
| [weeklyDiary/](weeklyDiary/) | Weekly reflections |
| [scripts/](scripts/) | Bayesian optimisation, logistic regression, neural networks, utils |

---

*This presentation summarises the BBO capstone as of the latest submissions and is intended for course assessment and portfolio use.*
