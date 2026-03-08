# Week 1 – Strategy (first week)

## Data and scope

- Week 1 is the **first** round: only **initial data** (no prior weekly submissions). The aim is to understand the data and choose the **first** query point for each function (for week 1 submission).

---

## Overall method and visualisation

- **Visualisation:** I only did basic data browsing and manual inspection: download the initial data and look at it (e.g. print inputs/outputs, simple plots). I had no shared plotting utilities yet (no 2D/3D scatter with distribution, no t-SNE, no results table). The second notebook was for deciding the first points by hand, not for automated analysis or comparison tables.

---

## What was done

- **Two notebooks:** (1) **`week1-initital_data_browsing.ipynb`** — I downloaded the initial data and visualised / explored it. (2) **`week1-deciding_first_points.ipynb`** — I analysed each function and suggested the most appropriate first point.
- **No model-based BO yet:** I chose points by **manual reasoning** and simple heuristics, not by a surrogate or acquisition function.
- **Function 1:** Centre point (0.5, 0.5) for simple exploration in a sparsely distributed space.
- **Function 2:** Correlation of high outputs around x₁ ≈ 0.65; chose (0.65, 0.5) to explore that region while keeping x₂ in an unexplored middle range.
- **Functions 3–8:** **Maximin sampling** — generate many random candidates, compute distance to existing inputs, and choose the point **furthest** from all others to maximise exploration of the input space.

---

## Why

- [weeklyDiary/week1.md](../../weeklyDiary/week1.md): “It’s the first week, and so I wanted to focus on exploration, picking values that were in uncertain regions.” High-dimensional functions were hard to visualise, so a distance-based maximin rule was used to pick points in unexplored space. The plan for the next round was to move to an algorithmic approach (e.g. Bayesian Optimization) once the first set of results was back.
