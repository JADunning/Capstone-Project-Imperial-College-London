# BBO Capstone Project — Presentation

**Imperial College London · Machine Learning Capstone · Black-Box Optimization**

---

This document presents an overview of the Black-Box Optimization (BBO) capstone project: objectives, strategy, evolution, insights, decision-making, and next steps. For setup, technical details, and project structure, see the **[README](README.md)**. For reproducibility and model documentation, see the **[Model Card](MODEL_CARD.md)** and **[Datasheet](DATASHEET.md)**.

---

## 1. Overview of the BBO Approach

### What we are trying to achieve

The goal is to **maximise the output** of eight unknown, expensive black-box functions with **very limited evaluations**—roughly one query per function per week over 10 weeks (~80 queries in total). We have no access to gradients or the internal form of the functions; we can only observe input–output pairs. The challenge is to use those observations strategically to find the best possible input configurations before the evaluation budget runs out.

In plain terms: we are trying to **discover where each “hidden” function is highest** by spending a small number of costly “experiments,” and to do this in a way that balances learning about the space (exploration) with capitalising on what we already know (exploitation).

### Overall strategy and how it fits together

The approach is **surrogate-based optimisation**: we fit a model (a “surrogate”) to the observed data and use it to decide where to query next. The strategy is **function-specific** rather than one-size-fits-all:

- **Function 1 (2D, sparse / near-binary):** Non-linear logistic regression learns a decision boundary between “good” and “bad” regions; we query near the boundary (e.g. where probability of “good” is ~0.5 or 0.9) to refine the best region.
- **Functions 2, 4, 5, 6 (2D–5D):** Gaussian process (GP) surrogate with acquisition functions (UCB or Expected Improvement). We use exploitative UCB (β = 0) to refine around promising areas, and switch to EI when progress stalls.
- **Function 3 (3D, noisy):** Noise-aware BO (GP with a WhiteKernel) and local restarts near the current best, to handle stochastic outputs.
- **Functions 7 and 8 (6D, 8D):** A PyTorch MLP is fitted to the data; the next point is suggested by gradient-based optimisation on this surrogate from multiple restarts.

Over time, the process moves from **early exploration** (weeks 1–3), through a **transition** (weeks 4–6), to **exploitation and refinement** (weeks 7–10). Validation—checking whether the latest submission improved on the previous best and how far suggested points are from the incumbent—guides where to focus each week.

---

## 2. How the Strategy Has Evolved

### Key changes since the early rounds

- **From uniform exploration to function-specific methods**  
  The first weeks used the same exploration-focused BO (GP + UCB, β = 2) for all functions, plus manual or maximin choices. As data accumulated, the strategy split by function: logistic regression for Function 1 (when its near-binary, boundary-like behaviour became clear), noise-aware BO for Function 3 (when noise was evident), and a neural-network-plus-gradient pipeline for the high-dimensional Functions 7 and 8 when GP became less practical.

- **From pure exploration to exploitation, then to EI when stuck**  
  After building initial surrogates, the approach shifted to exploitation (e.g. UCB with β = 0) to capitalise on promising regions. When exploitative BO stopped improving (e.g. drifting or plateauing), Expected Improvement was introduced to rebalance exploration and exploitation and to suggest points that could escape local optima.

- **From ad hoc inspection to structured validation**  
  Early analysis relied on simple 2D plots and manual inspection. This evolved into: (i) progress validation (did this week’s submission beat the previous best?), (ii) checking how far suggested points are from the current incumbent, and (iii) using surrogate diagnostics (e.g. MSE, length scales) to decide how much to trust or explore. t-SNE and other visualisations helped interpret higher-dimensional behaviour.

### What influenced these changes

- **Data trends:** Flat or sparse behaviour (Function 1), noisy outputs (Function 3), and high dimensionality (7, 8) directly drove the choice of surrogate and acquisition.
- **Feedback:** Week-on-week results showed when a method was not improving (e.g. exploitative BO suggesting points far from the best); that triggered switches (e.g. to EI or to a different method).
- **Model performance:** Low surrogate MSE increased confidence in exploitation; high MSE or unstable suggestions prompted more exploration or a different model (e.g. noise kernel for Function 3).
- **Intuition:** The idea that “one method per function” could reduce redundancy and match the problem (similar in spirit to focusing on high-variance directions in PCA) guided the move to specialised pipelines.

### Principles that guide query decisions now

- **Match the method to the function:** Sparse/binary → logistic regression; noisy → noise-aware GP; high-dimensional → NN + gradient; smooth and moderate-dimensional → GP + UCB/EI.
- **Validate progress:** Use tables and checks to see which functions improved and which stalled; focus tuning and method changes on stallers.
- **Exploit when the surrogate is trustworthy, explore when it isn’t:** Use surrogate quality (e.g. MSE, length scales) and distance-to-incumbent to decide how aggressively to exploit.
- **When exploitation stalls, rebalance:** Switch to EI (or similar) to allow exploration while still favouring improvement.

---

## 3. Patterns, Data and Insights

### Meaningful trends across data and evaluations

- **Function 1:** Most of the input space yields near-zero output; only a small region is “good.” The logistic boundary (especially with polynomial features) captures this structure; querying near the boundary (e.g. 0.5 or 0.9 probability contour) steadily refines the best region.
- **Function 3:** Outputs are noisy and often negative; repeated runs of the same or similar points can give different values. Noise-aware BO and occasional re-runs help avoid locking onto spurious peaks.
- **Functions 5, 7, 8:** Once a high-value region was found (e.g. Function 5 around ~0.9–0.95 in several dimensions), gains became incremental; the strategy naturally shifted to local refinement rather than global exploration.
- **Functions 2, 4, 6:** Mixed behaviour with local optima and varying smoothness; exploitative BO sometimes drifted. Introducing EI and progress validation helped refocus on regions that actually improved the incumbent.

### Variables and behaviours that influence results most

- **Function identity and dimensionality:** Performance and difficulty vary strongly by function; 2D is tractable with logistic regression and GP; 6D and 8D need the NN pipeline and are more sensitive to surrogate quality and restarts.
- **Noise (Function 3):** Noise level drives the need for a WhiteKernel and for not over-trusting single evaluations.
- **Strategy–problem match:** Using the right surrogate and acquisition for each function (logistic for 1, noise-aware for 3, NN for 7–8, GP+EI when UCB stalls) explains much of the variation in success.
- **Distance to incumbent:** When suggested points are close to the current best, we are in exploitation mode; when they are far, we are exploring. Tracking this distance helps interpret and debug suggestions.

### How these observations shape understanding of the search process

The search is best thought of as **iterative and adaptive**: early data reveal function “type” (sparse, noisy, unimodal, multi-modal, high-dimensional), which then dictates the surrogate and acquisition. Variance in outcomes is largely explained by **function type** and **strategy–problem fit** rather than by a single global recipe. Focusing on the dimensions or directions that the surrogate says matter most (e.g. logistic coefficients, GP length scales) is analogous to retaining the principal components that explain the most variance—it keeps the essential structure and reduces redundant or noisy dimensions when deciding where to query next.

---

## 4. Decision-Making and Iteration

### Balancing exploration and exploitation

- **Early rounds (1–3):** Emphasis on exploration—diverse sampling, maximin or exploratory UCB—to build surrogates and discover promising regions.
- **Middle and later rounds (4–10):** Shift to exploitation (e.g. UCB β = 0, local refinement) to improve the incumbent. When progress stalls or suggestions drift far from the best, we increase exploration again via EI or by re-checking surrogate assumptions.
- **Ongoing checks:** Progress validation and distance-to-incumbent tables indicate whether we are over-exploiting (no improvement) or over-exploring (suggestions too far from the best); that informs the next week’s choice of acquisition or method.

### Examples of strategic decisions

- **Function 1 — switching from BO to logistic regression:**  
  BO was suggesting extreme points (e.g. corners) because the function is mostly flat with a small “good” region. **Decision:** Use logistic regression to learn the boundary and query near it. **Result:** More stable improvement and suggestions that stay in the relevant region. This worked because it matched the problem structure.

- **Exploitative BO drifting or plateauing:**  
  For some functions, UCB with β = 0 kept suggesting points that did not improve the best observed value or that were far from the incumbent. **Decision:** Introduce Expected Improvement and progress validation; use EI when the incumbent had not improved. **Result:** Better balance and a way to escape local behaviour. What didn’t work was continuing with pure exploitation without a formal check for progress.

### Handling uncertainty and unexpected results

- **Unexpectedly bad or noisy outcomes:** If a suggested point performs poorly or very differently from the surrogate’s prediction, we treat it as new data and refit; for Function 3, we explicitly allow for noise in the model and sometimes re-run points. We do not assume the surrogate is perfect.
- **When results don’t match expectations:** We use validation (did we improve? how far is the suggestion from the best?) and surrogate diagnostics (MSE, length scales). If the surrogate is unreliable or progress stalls, we change method or acquisition (e.g. switch to EI, or revisit hyperparameters) rather than repeating the same strategy.

---

## 5. Next Steps and Reflection

### Planned actions to improve performance

- **Final round (e.g. Module 24):** With only one round left, the plan is a final exploitation pass: use the current best method per function and try to refine further around the best-known points, with no major new exploration.
- **If the project were to continue:** (i) Tighten NN and BO settings (e.g. more epochs, restarts, or kernel choices) where compute allows; (ii) formalise progress validation (e.g. automatic “switch to EI if no improvement for k weeks”); (iii) consider regression or meta-analysis across functions to prioritise which function to tune each week.

### Connection to the broader ML landscape

This project sits at the intersection of **Bayesian optimisation**, **surrogate-based optimisation**, and **experimental design**: we use probabilistic models (GP, logistic, NN) to approximate expensive objectives and acquisition functions to decide the next experiment. The same ideas underpin hyperparameter tuning (e.g. Optuna, SMAC), materials discovery, and drug design, where each evaluation is costly and we must learn from limited data. The capstone highlights that **method choice and exploration–exploitation balance** are as important as the surrogate itself, and that **documentation and validation** (model cards, datasheets, progress checks) support reproducibility and robust decision-making in real-world optimisation.

### Communicating results to a non-technical audience

To a stakeholder or non-technical audience, we would say: *“We had a limited number of expensive ‘experiments’ on eight different unknown systems. We used the results of past experiments to build simple predictive models and then chose each new experiment to either learn more about the space or to improve on our best result so far. We tailored the approach to each system—for example, we used a different strategy for noisy or high-dimensional systems—and we kept track of whether we were actually improving. The main takeaway is that we improved over time by combining exploration and exploitation and by adapting the method to what the data showed about each system.”*

---

## Quick Links

| Document | Purpose |
|----------|---------|
| [README](README.md) | Project overview, setup, inputs/outputs, technical approach |
| [MODEL_CARD.md](MODEL_CARD.md) | Optimisation strategy, intended use, performance, limitations |
| [DATASHEET.md](DATASHEET.md) | Dataset motivation, composition, collection, preprocessing |
| [weeklyDiary/](weeklyDiary/) | Weekly reflections and strategy notes |
| [scripts/](scripts/) | Bayesian optimisation, logistic regression, neural networks, utils |

---

*This presentation summarises the BBO capstone as of the latest submissions and is intended for course assessment and portfolio use.*
