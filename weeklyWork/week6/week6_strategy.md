# Week 6 – Strategy (differences from Week 5)

## Data and scope

- I now load **weeks 1–5** (week 5 data added); next-point recommendations are for **week 6** submission.

---

## Overall method and visualisation

- **Visualisation:** I kept 2D/3D scatter and **t-SNE** in the BO notebooks (function2_4_5_6 and function3). For **Functions 7 and 8** (PyTorch with gradient optimization) I added **t-SNE** there too, so I could visualise high-dimensional inputs and the suggested next point. I introduced **progress validation** (e.g. checks if I'm improving, flags if stuck) in the BO workflow, but I still did not have a single **results table** with columns like Source, Output value, **Distance to maximum input**, and **Summary of progress**; that comes in week 8.

---

## Functions 2, 3, 4, 5 and 6

- **Change:** I **split out** Function 3 from the combined week 5 notebook. Week 5 I had one notebook for Functions 2, 3, 4, 5 and 6 (`function2_3_4_5_6_bayesian_exploitation.ipynb`). Week 6 I have **`function2_4_5_6_bayesian_validation.ipynb`** (Functions 2, 4, 5, 6) and **`function3_bayesian_validation.ipynb`** (Function 3 only).
- **Naming:** I renamed the shared BO notebook from “exploitation” to **“validation”** and it now includes **progress validation** (checks if we’re improving, flags if stuck).
- **Why:** Function 3’s objective appears noisy; isolating it in its own notebook (as in the week 6 F3 intro) makes it easier for me to inspect the data and try a different Bayesian approach without coupling that work to the other functions. Adding progress validation supports the layered approach described in [weeklyDiary/week6.md](../../weeklyDiary/week6.md): exploration → exploitation → automatic progress validation, and helps flag when exploitation plateaus.

---

## Functions 7 and 8

- **Change:** I switched next-point suggestion for the PyTorch MLP surrogate from **random sampling** (week 5: sample candidates in [0, 1]) to **gradient-based optimization** (week 6: gradient ascent from multiple restarts to find the maximum of the NN).
- **Why:** Gradient-based optimization is more efficient and finds better optima than random sampling (as in week 6 diary: “random sampling (50,000 evaluations) is thorough but slower, while gradient optimisation (~2,500 evaluations) is more efficient and finds better optima”). The NN’s gradients are used to climb to the optimum, similar to using backpropagation in a CNN.
