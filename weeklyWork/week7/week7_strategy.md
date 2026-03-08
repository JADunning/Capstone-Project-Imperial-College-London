# Week 7 – Strategy (differences from Week 6)

## Data and scope

- I now load **weeks 1–6** (week 6 data added) in all notebooks; next-point recommendations are for **week 7** submission.

---

## Overall method and visualisation

- **Visualisation:** I continued to use 2D/3D scatter and **t-SNE** in the BO notebooks (function2_4_6, function5) and in function7_8 (PyTorch). Function 3 (next evaluation) used **3D scatter** only (3D input space). I still did not introduce the **results table** with distance to the best input and progress summary; that appears in week 8. Analysis was a mix of plots and progress-validation logic in code.

---

## Functions 2, 4, 5 and 6

- **Change:** I **split out** Function 5 from the combined week 6 notebook. Week 6 had one notebook for Functions 2, 4, 5 and 6 (`function2_4_5_6_bayesian_validation.ipynb`). Week 7 I have **`function2_4_6_bayesian_validation.ipynb`** (Functions 2, 4, 6 only) and **`function5_next_evaluation.ipynb`** (Function 5 only).
- **Why:** Giving Function 5 its own notebook gives me a dedicated “next evaluation” workflow (incumbent, progress validation, BO-suggested next point and save to results) and keeps the 2–4–6 notebook focused. Progress validation (as in [weeklyDiary/week7.md](../../weeklyDiary/week7.md)) is easier when each function has its own pipeline so I can re-check the surrogate on functions that are not improving.

---

## Function 3

- **Change:** I refocused the notebook from **“bayesian_validation”** to **“next evaluation”**: **`function3_next_evaluation.ipynb`** loads all observed data, identifies the incumbent, and defines my approach for the next evaluation (week 7 submission).
- **Why:** I split out work on Function 3 because I wasn’t getting good results; a dedicated next-evaluation notebook keeps the workflow clear and lets me iterate on modelling choices (e.g. noise-aware BO) for this noisy function without coupling to the other functions.
