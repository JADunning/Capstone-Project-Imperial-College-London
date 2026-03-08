# Week 2 – Strategy (differences from Week 1)

## Data and scope

- Week 2 uses **initial data plus week 1** results; next-point recommendations are for **week 2** submission.

---

## Overall method and visualisation

- **Visualisation:** With the move to BO I introduced the full visualisation suite in one notebook: **2D scatter** and **3D scatter** (with distribution), **t-SNE** for high-dimensional functions (so I could see structure in 4D–8D), **BO visualization** (2D plot with the BO-suggested point), and **function summary subplot** (overview across functions). So **t-SNE first appears in week 2**, alongside the single BO pipeline. I did not yet have a results table with distance to the best point or progress summary; analysis was mainly by looking at these plots.

---

## Approach

- **Change:** Week 1 I had **no** Bayesian Optimization: I downloaded and visualised the initial data, then in a separate notebook I chose the first points by hand (e.g. centre (0.5, 0.5) for F1, correlation-based pick for F2, **maximin sampling** for the rest—pick the point furthest from existing points). Week 2 I introduced a **single BO pipeline for all functions**: one notebook (`week2.ipynb`) with a simple **Gaussian Process surrogate** and **UCB acquisition**, used for every function.
- **Why:** In week 1 I wrote: “Maybe an algorithmic approach could work well like Bayesian Optimisation to help pick the next steps. So I’d probably work on getting that up and running next week.” Week 2 I wanted to move to a BO process instead of manual interpretation, still focusing on **exploration** because it’s early and you only have 11 data points; the BO classes influenced my plan to use at least 3 weeks on exploration before exploiting.

---

## What stayed the same

- **Exploration focus:** I did not switch to exploitation; the same BO is used for all functions with an exploration-oriented setting so that week 2 continues to sample uncertain or unexplored regions rather than targeting known high performers.
