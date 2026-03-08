# Week 11 – Strategy (differences from Week 10)

## Data and scope

- All notebooks now load **weeks 1–10** (added week 10 data from `data/week10`). Week 10 notebooks used weeks 1–9.

## Function 1 (non-linear logistic regression)

- **Target probability:** Configurable target contour; this week we use `TARGET_PROB = 0.9` (week 10 used 0.8). Next point is chosen on the 90% contour, with local optimization from the incumbent so the suggestion stays near the best region.
- **Plot fix:** `plot_2d_scatter_with_distribution` is called with a single “new” point (week 10 only); `weekly_points` still lists all weeks 1–10.
- **Class-assignments table (Function 1):** Added **Rank** (1 = best) and **Size** bar for quick eyeballing; dropped the log10 column; clarified that the threshold is the chosen percentile (Class 1 = top %). Makes it easier to see relative output size at a glance.

## Naming (from week 10)

- **function2:** `function2_bayesian_validation` → `function2_local_trust_region_bo.ipynb` (local trust-region + duplicate aggregation).
- **function3:** `function3_comparison_rerun` → `function3_noise_aware_exploitative_bo.ipynb` (optimisation-only, noise-aware exploitative BO).
- **function5:** `function5_next_evaluation` → `function5_local_trust_region_bo.ipynb` (same BO as function 2).

## Unchanged from week 10

- Function 2, 3, 4, 5, 6, 7, 8 methods and BO scripts are unchanged; only data scope (through week 10), plot call fixes, and notebook filenames differ.
- Regression focus TODO (week 6) still applies; see week 10 notes if needed.
