# Model Card: BBO Optimisation Approach

This model card describes the Black-Box Optimization (BBO) approach used in the Imperial College London ML Capstone project, following the framework from Mini-lesson 21.2. It builds on reflections from the weekly diaries and required components.

---

## 1. Overview

- **Name**: BBO Capstone Optimisation Strategy (function-specific surrogates and acquisition).
- **Type**: Suite of surrogate-based optimisation methods; not a single fixed “model” but a **strategy** that selects different techniques per function and evolves over the 10 rounds.
- **Version**: As of the latest submission (e.g. week 8–10): logistic regression (Function 1), GP + UCB/EI (Functions 2, 4, 5, 6), noise-aware GP (Function 3), PyTorch MLP + gradient optimisation (Functions 7, 8). Versioning is implicit in the weekly notebooks and `scripts/` (e.g. `bayesian_optimization_ei.py`, `bayesian_optimization_exploitative.py`, `bayesian_optimization_noise.py`, `pytorch_simple_nn_next_point.py`).

---

## 2. Intended Use

**Suitable tasks**

- **Maximising expensive black-box functions** with very few evaluations (on the order of ~10 per function).
- **Functions with inputs in [0, 1]^d** and scalar outputs, where d is 2–8 and matches the chosen method (2D for logistic regression; 2–6D for GP; 6–8D for NN).
- **Educational and capstone use**: demonstrating exploration vs exploitation, surrogate choice, and the impact of noise and dimensionality.

**Use cases to avoid**

- **High-dimensional spaces** (e.g. d >> 8) without adaptation: GP becomes expensive; the current NN setup is modest and may need scaling (week 9).
- **Strongly non-smooth or discontinuous objectives**: The approach assumes surrogates (especially GP with RBF kernel) approximate the objective reasonably well; highly jagged or discontinuous functions may not be handled well (week 10 assumption).
- **Production deployment without validation**: The strategy is tuned for this challenge (limited queries, specific bounds); different budgets or constraints would require re-evaluation and possibly different methods.
- **Optimisation where the optimum is outside [0, 1]^d**: The strategy is bounded; it will not find optima outside the stated box (week 10).

---

## 3. Details: Strategy Across the Ten Rounds

**High-level evolution**

- **Rounds 1–3 (exploration)**: Broad sampling across the input space to build initial surrogates and understand function behaviour.
- **Rounds 4–6 (transition)**: More exploitation (e.g. β=0 UCB), introduction of progress validation (flagging when the incumbent does not improve), noise-aware BO for Function 3, and gradient-based next-point suggestion for Functions 7 and 8.
- **Rounds 7–10 (exploitation and refinement)**: Focus on refining around best-known regions; for functions where exploitative BO drifted or plateaued, switch to **Expected Improvement (EI)** to balance exploration and exploitation (week 10). Continued use of logistic regression (Function 1), noise-aware BO (Function 3), and NN + gradient (Functions 7, 8).

**Techniques by function**

| Function | Dimensions | Technique | Main ingredients |
|----------|------------|-----------|-------------------|
| 1 | 2D | Non-linear logistic regression | Polynomial features, binary labels (threshold percentile), grid search over degree/C/threshold; query where P(good) ≈ 0.5. |
| 2, 4, 5, 6 | 2D–5D | Bayesian optimisation (GP) | GP (RBF kernel), UCB with β=0 (exploitation) or EI when progress stalls; progress validation to refocus. |
| 3 | 3D | Noise-aware BO | GP + WhiteKernel, moderate β, local restarts near incumbent; re-runs for consistency. |
| 7, 8 | 6D, 8D | Neural network + gradient | PyTorch MLP (e.g. 64 hidden, 1200 epochs), fit to observations; gradient ascent from multiple restarts to suggest next point. |

**Implementation**

- **Scripts**: `scripts/bayesian_optimisation/` (EI, exploitative, noise-aware), `scripts/logistic_regression/`, `scripts/neural_networks/pytorch_simple_nn_next_point.py`, `scripts/utils/data_utils.py`.
- **Notebooks**: Weekly analysis and next-point generation in `weeklyWork/weekN/` (e.g. function1_nonlinear_logistic_regression, function2_bayesian_validation, function7_8_pytorch_gradient_optimization). Results (e.g. next point, method name, hyperparameters) are stored in `weeklyWork/weekN/results/function_k_next_point.json`.

---

## 4. Performance

**Scope**

- Performance is measured **per function** over the eight black-box functions, using the **maximum observed output** (and, in reflection, distance to the best point and week-on-week progress).

**Metrics**

- **Primary**: Maximum output value observed so far for each function (higher is better; all functions are maximisation).
- **Secondary**: Progress summaries (whether the latest submission improved on the previous best), and Euclidean distance of suggested or chosen inputs to the current incumbent (used for analysis in week 10).
- No formal train/test split; the “evaluation” is the live weekly submission. Internal metrics (e.g. surrogate MSE, acquisition value) are used for tuning and diagnostics in notebooks.

**Summary across the eight functions**

- Performance varies by function and by round. Functions 1 and 3 have particular structure (sparse/near-binary; noisy), so improvements can be uneven. Functions 5 and 7–8 show diminishing returns as the strategy approaches local/global optima (week 9). The combination of method choice (logistic vs GP vs NN), acquisition (UCB vs EI), and progress validation is intended to improve robustness and steady gains where possible (week 9, 10).

---

## 5. Assumptions and Limitations

**Assumptions (from week 10 and earlier reflections)**

- **Smoothness**: The objective is smooth enough to be approximated by a **GP with an RBF kernel** (for Functions 2–6 and the noise-aware variant for 3). If the true function is highly jagged, discontinuous, or has very different length scales per dimension, the surrogate can be poor and suggestions may drift or get stuck.
- **Bounded optimum**: The global optimum is assumed to lie **within [0, 1]^d**; the strategy does not search outside this box.
- **NN as proxy (Functions 7, 8)**: The neural network surrogate is assumed to be a reasonable proxy for the black-box in the regions of interest; if it generalises poorly in under-sampled areas, suggested points may be suboptimal.
- **Limited compute**: Modest NN and BO settings (e.g. 1200 epochs, 64 hidden, 25 restarts) are used to keep runtime manageable on a laptop; this can limit surrogate quality (week 9).

**Limitations and failure modes**

- **One submission per function per week**: Introduces temporal sparsity and limits ability to re-sample or explore densely; strategy must be robust to this delay (week 10).
- **Time and focus**: In practice, only 2–3 functions are deeply optimised each week; others may receive less tuning, which can bias outcomes (week 10).
- **Overfitting on small data**: With ~10–17 points per function, surrogates (especially flexible ones) can overfit; validation and progress checks help but do not remove the risk (week 7).
- **Noise**: Function 3 is explicitly noisy; the noise-aware BO and re-runs mitigate but do not eliminate the risk of locking onto spurious peaks.

---

## 6. Ethical Considerations

**Transparency and reproducibility**

- **Documentation**: This model card and the **datasheet** (DATASHEET.md) make the task, data composition, collection process, and strategy explicit. Together they support reproducibility: others can see what data were used, how queries were generated, and what assumptions and limitations apply.
- **Repository structure**: Weekly notebooks, `scripts/`, and `results/` allow others to follow the logic and, with the same initial and weekly data, re-run the pipeline and understand why certain points were chosen (as in week 10 reflection on transparency).
- **Reflections**: Weekly diaries and required components (e.g. 21.1) document reasoning, assumptions, and gaps; linking these in the model card and datasheet improves clarity and accountability.

**Real-world adaptation**

- Making **intended use**, **limitations**, and **failure modes** explicit helps anyone adapting this approach to a different domain (e.g. hyperparameter tuning, drug discovery) to assess fit and risk. Stating assumptions (smoothness, bounded domain, surrogate validity) and inappropriate uses reduces misuse and overconfidence.

