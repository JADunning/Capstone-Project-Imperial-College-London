# Week 5 – Strategy (differences from Week 4)

## Data and scope

- I now load **weeks 1–4** (week 4 data added); next-point recommendations are for **week 5** submission.

---

## Overall method and visualisation

- **Visualisation:** For the BO notebook (functions 2–6) I used 2D scatter, 3D scatter, and **t-SNE** as before. For **Function 1** (non-linear logistic regression) I kept **2D scatter** only (and probability contours). For **Functions 7 and 8** (PyTorch) I did not add t-SNE or the full scatter suite in week 5; that notebook was focused on fitting the NN and sampling the next point. I still had no table of results with distance to the best point or progress summary.

---

## Function 1

- **Change:** I moved Function 1 from **linear** logistic regression (week 4: `function1_logistic_regression.ipynb`) to **non-linear** logistic regression (week 5: `function1_nonlinear_logistic_regression.ipynb`) with polynomial features.
- **Why:** Week 4 diary noted that Function 1 might need a **non-linear** (curved) decision boundary; building on the linear model from week 4, the non-linear version adds complexity incrementally (as in [weeklyDiary/week5.md](../../weeklyDiary/week5.md): “from simple linear logistic regression … now added in a more complex non-linear regression”). This matches the idea of hierarchical feature learning: coarse then finer structure.

---

## Functions 2, 3, 4, 5 and 6

- **Change:** I added a new notebook **`function2_3_4_5_6_bayesian_exploitation.ipynb`** introduces **exploitative Bayesian Optimization** (GP + UCB with β=0) for all of these functions. Week 4 I had no dedicated BO notebook for 2–6 (only the general `week4.ipynb` for data and visualisation).
- **Why:** Early iterations explored broadly; week 5 I moved to **exploitation** and refining (week 5 diary: “early iterations explored broadly, now i'm moving more to exploitation and refining”). Pure exploitation (β=0) targets promising regions identified during exploration.

---

## Functions 7 and 8

- **Change:** I added a new notebook **`function7_8_pytorch_next_point.ipynb`** introduces a **PyTorch MLP** surrogate for the high-dimensional functions: fit the NN to observed data, then suggest the next point by **sampling candidate points** in [0, 1] and taking the one with the highest predicted value.
- **Why:** For 6D and 8D, a neural network surrogate is more scalable than a single GP; week 4 diary I planned to “look at training a simple neural network for some of the higher dimension functions slightly later on.” Week 5 I implemented that step with a small MLP and random sampling for the next point.
