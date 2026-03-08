# Week 8 – Strategy (differences from Week 7)

## Data and scope

- I now load **weeks 1–7** (week 7 data added) in all notebooks; next-point recommendations are for **week 8** submission.

---

## Overall method and visualisation

- **Visualisation:** I brought in the **results table** that shows, for each observation: **Source** (e.g. initial, week1, …), **Input number**, **Output value**, **Distance to maximum input** (Euclidean distance to the input with the highest output so far), and **Summary of progress** (e.g. Improving / Not improving vs the previous week). I use `make_inputs_outputs_table` from `scripts.utils.distance_utils` so the same table can be reused across functions and weeks. This is when I started systematically comparing how far each point is from the current best and whether the last submission improved. I kept 2D/3D scatter and **t-SNE** where relevant (e.g. function2, function4_6, function5, function7_8); function3 (comparison at incumbent) uses 3D scatter.

---

## Functions 2, 4 and 6

- **Change:** I **split out** Function 2 from the combined week 7 notebook. Week 7 had a single notebook for Functions 2, 4 and 6 (`function2_4_6_bayesian_validation.ipynb`). Week 8 I have a **dedicated notebook for Function 2** (`function2_bayesian_validation.ipynb`) and a separate notebook for 4 and 6 (`function4_6_bayesian_validation.ipynb`).
- **Function 2 only:** The new Function 2 notebook uses the **mean-only exploitative** BO module (`bayesian_optimization_exploitative.py`): next point by **maximizing GP mean** (no EI), and the proposed row uses the **GP predicted mean** as the output value so it stays on the same scale as observed y. Functions 4 and 6 keep the same exploitative BO (GP + UCB β=0) as in week 7.
- **Why:** Giving Function 2 its own notebook gives me function-specific handling and a clear “mean-only” setup: the proposed row is the raw GP mean at the suggested point instead of UCB, which keeps the validation table on a consistent scale and makes it easier to spot implausible suggestions. The split also supports progress validation per function (as in [weeklyDiary/week7.md](../../weeklyDiary/week7.md)), so I can re-check the surrogate on functions that are not improving.

---

## Function 3

- **Change:** I added a new notebook **`function3_comparison_rerun.ipynb`** (replacing the week 7 “next evaluation” style). The focus is on **comparing two results at the incumbent**: the first time that point was observed (Initial or an earlier week) and the **Week 7 re-run** at the same point. I set up visualisation and tables to show both.
- **Why:** Function 3 is noisy; I wasn't getting good results so I split out work on Function 3. Comparing the first observation and the Week 7 re-run at the incumbent lets me separate noise from real change and decide whether to trust the surrogate or adjust the strategy (e.g. noise-aware BO, as in week 7).
