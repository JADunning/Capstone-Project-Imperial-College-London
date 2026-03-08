# Week 9 – Strategy (differences from Week 8)

## Data and scope

- I now load **weeks 1–8** (week 8 data added) in all notebooks; next-point recommendations are for **week 9** submission.

---

## Overall method and visualisation

- **Visualisation:** I kept the same structure as week 8: the **results table** (Source, Input number, Output value, Distance to maximum input, Summary of progress) in each function notebook, plus 2D/3D scatter and **t-SNE** where the input dimension or notebook focus made it useful. No new visualisation or analysis format this week; the change was in the BO method (exploitative_2) and data scope, not in how I visualise or tabulate results.

---

## Function 2

- **Change:** I switched from `bayesian_optimization_exploitative.py` to `bayesian_optimization_exploitative_2.py`.
- **What’s different:** (1) Duplicate and near-duplicate inputs are **aggregated** before fitting the GP (cluster centre and mean output). (2) A **WhiteKernel** is added so the GP models observation noise. (3) The acquisition step **maximizes the GP mean only in a local trust region** around the incumbent instead of scanning the whole space. (4) **Output clipping** is removed; the proposed row uses the raw GP predicted mean.
- **Why:** The previous exploitative BO was less effective on Function 2: treating repeated or very similar evaluations as separate points created spurious peaks in the GP; the lack of a noise term made the mean overconfident; global maximization chased those artefacts and suggested points far from the best; clipping only hid implausible predictions. The new algorithm keeps suggestions near the incumbent and handles repeated/noisy evaluations in a stable way (as reflected in week 11 diary).

---

## Function 3

- **Change:** I extended data scope through **week 8**; comparison at incumbent now uses the **Week 8** evaluation at the incumbent (week 8 notebook used the Week 7 re-run).
- **Why:** To keep the comparison-at-incumbent check in line with the latest submission and to continue monitoring consistency at the best point for this noisy function.

---

## Function 5

- **Change:** I switched from `bayesian_optimization_exploitative` to `bayesian_optimization_exploitative_2.py` — same as Function 2: noise-aware GP, local trust-region exploitation, duplicate aggregation, raw GP mean for the proposed row.
- **Why:** Same rationale as Function 2: more robust behaviour with repeated or similar evaluations and fewer spurious suggestions. Week 9 diary also emphasises favouring robust, interpretable methods and re-checking local optima on noisy functions (e.g. 3 and 5), which this pipeline supports.

