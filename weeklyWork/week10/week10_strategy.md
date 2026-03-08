# Week 10 – Strategy (differences from Week 9)

## Data and scope

- I now load **weeks 1–9** (week 9 data added) in all notebooks; next-point recommendations are for **week 10** submission.

---

## Overall method and visualisation

- **Visualisation:** I continued to use the **results table** (Source, Input number, Output value, Distance to maximum input, Summary of progress) and 2D/3D scatter and **t-SNE** as in previous weeks. The week 10 diary describes the table I made comparing output, Euclidean distance to the maximum, and progress summary to see where I was improving and to focus effort on functions that weren’t; that workflow relies on this same table and distance-based view. No new visualisation types this week.

---

## Function 2

- **Change:** Notebook renamed to `function2_local_trust_region_bo.ipynb` (week 9 used `function2_bayesian_validation.ipynb` for the same strategy).
- **Why:** To reflect that this notebook uses my improved BO (local trust-region + duplicate aggregation + WhiteKernel). The previous exploitative BO was less effective on Function 2: no duplicate handling and no observation noise led to spurious GP spikes and unstable suggestions; global maximization chased those artefacts. I adopted the new algorithm to fix that; the rename makes it clear which approach the notebook implements.

---

## Function 3

- **Change:** New notebook `function3_noise_aware_exploitative_bo.ipynb` — optimisation-only, noise-aware exploitative BO; comparison / re-run workflow removed. `function3_comparison_rerun.ipynb` remains for reference (week 9’s comparison-at-incumbent style).
- **Why:** To simplify my workflow and match the structure I use for functions 4 and 6, so Function 3 is driven by a single optimisation pipeline. I kept the comparison/rerun notebook for reference when analysing incumbent consistency.

---

## Function 5

- **Change:** New notebook `function5_local_trust_region_bo.ipynb` for the same pipeline (week 9 used `function5_next_evaluation.ipynb`).
- **Why:** Same rationale as Function 2: the name reflects the switch to local trust-region + duplicate aggregation + noise-aware GP, so it’s clear which BO strategy the notebook uses.
