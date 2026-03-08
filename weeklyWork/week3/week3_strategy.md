# Week 3 – Strategy (differences from Week 2)

## Data and scope

- I now load **weeks 1–2** (week 2 data added); next-point recommendations are for **week 3** submission.

---

## Overall method and visualisation

- **Visualisation:** I kept the same approach as week 2: one notebook with 2D scatter, 3D scatter, **t-SNE**, BO visualization, and function summary subplot for all functions. No new visualisation types; no results table with distance or progress. Analysis was still mainly plot-based.

---

## Approach

- **Change:** I kept the **same** overall setup as week 2: one main notebook (`week3.ipynb`) that loads initial data and week 1–2 data, combines it, and uses the same **Bayesian Optimization (GP + UCB)** for all functions. No new function-specific notebooks; no switch away from exploration.
- **Why:** Week 3 diary: “My strategy hasn’t changed at all, it’s week 3 so I’m sticking with exploration focused Bayesian Optimisation. This is part of the original strategy, to focus on exploration for a few weeks before then exploring other alternatives.” So week 3 is still exploration-focused BO for all functions, with one more week of data.

---

## Function 1 (observation that will drive week 4)

- Week 3 diary notes that **Function 1** is struggling: the function is very flat and the BO is tending to extremes (e.g. (0,0)); I manually picked (0.6, 0.6) near the transition and plan to “code a logistic regression next week.” That sets up the week 4 change (dedicated F1 logistic regression notebook).
