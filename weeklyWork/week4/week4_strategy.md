# Week 4 – Strategy (differences from Week 3)

## Data and scope

- I now load **weeks 1–3** (week 3 data added); next-point recommendations are for **week 4** submission.

---

## Overall method and visualisation

- **Visualisation:** In the general `week4.ipynb` I kept 2D/3D scatter, **t-SNE**, BO visualization, and function summary subplot for the functions that still used the shared BO. The new **Function 1** notebook only uses **2D scatter with distribution** (no t-SNE there), which is enough for the 2D input space. I still did not have a structured results table (e.g. distance to maximum input or progress summary); analysis remained plot-based and per-notebook.

---

## Function 1

- **Change:** I added a dedicated notebook **`function1_logistic_regression.ipynb`**: fit a **logistic regression** to Function 1 data, tune hyperparameters (e.g. by MSE), and use the best model to choose the next point **closest to the decision boundary** (where P(Class 1) ≈ 0.5). Week 3 I had only the single `week3.ipynb` with no function-specific surrogate for F1.
- **Why:** Week 3 I noted that the first function was struggling with the shared BO (flat function, BO tending to extremes like (0,0)) and that I would “move to logistic regression” and “code a logistic regression next week.” Function 1 has a clear two-class structure ([weeklyDiary/week4.md](../../weeklyDiary/week4.md): “data that sits in 2 categories”); framing BBO as classification (good vs bad by a threshold) and querying near the boundary focuses exploration where it matters.

---

## General (week4.ipynb)

- **Change:** I kept a general **`week4.ipynb`** for loading data (initial + weeks 1–3), combining it, and visualisation (e.g. `plot_function_summary_subplot`), similar in role to week 3’s single notebook but with an extra week of data and the new F1 notebook alongside it.
