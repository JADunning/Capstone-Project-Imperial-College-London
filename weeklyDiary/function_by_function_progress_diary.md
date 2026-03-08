# Function-by-Function Progress Diary

This document tracks the optimization strategy and progress for each function across weeks 1-4.

---

## Function 1 (2D - Maximize)

**Initial Data:** 10 samples, 2 dimensions  
**Description:** Detect contamination sources in a 2D area (radiation field detection)

- **Week 1:** Manual selection based on visualization - chose center point (0.5, 0.5) for simple exploration in sparsely distributed space. **Output:** 2.6752879910742468e-09
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - exploration-focused BO suggested corner point (0.000000, 0.000000) balancing exploration/exploitation. **Output:** 2.308810742126141e-248
- **Week 3:** Manual override of BO - BO suggested same point as Week 2, manually adjusted to (0.6, 0.6) near transition region because function is very flat and BO was converging to extremes. **Output:** 0.025559285339829783
- **Week 4:** Started implementing simple linear logistic regression. The output and categorisation seemed to show that maybe it's not a linear logistic regression. Might try something else next week!




Week 11
- For this week, we target `TARGET_PROB = 0.9` (instead of only the 0.5 boundary) and optimize from the incumbent so the suggestion stays near the best-known region.
- Class-assignments table: added **Rank** (1 = best) and **Size** bar for quick eyeballing of relative output size; dropped the confusing log10 column and clarified that the threshold is the chosen percentile (Class 1 = top %).



---

## Function 2 (2D - Maximize)

**Initial Data:** 10 samples, 2 dimensions  
**Description:** Black-box ML model optimization with noisy outputs and local optima

- **Week 1:** Manual selection based on output correlation - identified correlation of high outputs around x1=0.65, selected (0.65, 0.5) to explore promising region while maintaining exploration in x2. **Output:** 0.42721035515778233
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - exploration-focused BO suggested boundary point (0.839041, 1.000000) guiding toward uncertain/promising regions. **Output:** 0.09143573616583116
- **Week 3:** Bayesian Optimization (GP + UCB) - continued exploration, BO refining search based on accumulated observations, suggested (0.763745, 0.848033). **Output:** 0.3317119096486708
- **Week 4:** Bayesian Optimization (GP + UCB) - continued exploration-focused approach, BO learning from growing dataset. **Output:** (BO suggestion from week 4 run)

---

## Function 3 (3D - Maximize)

**Initial Data:** 15 samples, 3 dimensions  
**Description:** Drug discovery - optimizing three compound combinations to minimize adverse reactions

- **Week 1:** Maximin sampling - generated 5000 random candidates, selected (0.466753, 0.990442, 0.996265) maximizing minimum distance to existing points for exploration of unexplored 3D regions. **Output:** -0.45300818056360315
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - exploration-focused BO suggested boundary point (1.000000, 0.000000, 0.757830) guiding search in 3D space. **Output:** -0.16648410423830667
- **Week 3:** Bayesian Optimization (GP + UCB) - continued exploration, BO exploring boundary regions, suggested (1.000000, 1.000000, 0.000000). **Output:** -0.16364425027712573
- **Week 4:** Bayesian Optimization (GP + UCB) - continued exploration-focused approach in 3D space. **Output:** (BO suggestion from week 4 run)

---

## Function 4 (4D - Maximize)

**Initial Data:** 30 samples, 4 dimensions  
**Description:** Warehouse product placement optimization with four hyperparameters

- **Week 1:** Maximin sampling - generalized maximin function selected (0.013827, 0.388082, 0.736064, 0.971728) farthest from existing 30 points for exploration of unexplored 4D regions. **Output:** -25.81769735182645
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - exploration-focused BO suggested (0.413267, 0.391233, 0.327773, 0.428435) in 4D space. **Output:** 0.07635232865804342 (first positive output!)
- **Week 3:** Bayesian Optimization (GP + UCB) - BO refining search around promising region from Week 2, suggested (0.434166, 0.428579, 0.217539, 0.429707) near previous point. **Output:** -2.443739075344467
- **Week 4:** Bayesian Optimization (GP + UCB) - continued exploration, BO learning from accumulated data. **Output:** (BO suggestion from week 4 run)

---

## Function 5 (4D - Maximize)

**Initial Data:** 20 samples, 4 dimensions  
**Description:** Chemical process yield optimization - typically unimodal with single peak

- **Week 1:** Maximin sampling - generalized maximin function selected (0.937239, 0.959791, 0.960767, 0.818897) farthest from existing 20 points for exploration. **Output:** 4219.157963029058 (very high output!)
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - BO identified high-value region from Week 1, suggested (0.922382, 0.960497, 0.963203, 0.824728) very close to Week 1 point for nearby exploration. **Output:** 4139.609565498906 (high but slightly lower)
- **Week 3:** Bayesian Optimization (GP + UCB) - BO refining search around identified high-performance region, suggested (0.939030, 0.958851, 0.959387, 0.818091) in same high-value region. **Output:** 4205.41112909808 (high output maintained)
- **Week 4:** Bayesian Optimization (GP + UCB) - BO continuing to explore/exploit high-value region. **Output:** (BO suggestion from week 4 run)

---

## Function 6 (5D - Maximize)

**Initial Data:** 20 samples, 5 dimensions  
**Description:** Cake recipe optimization - five ingredients, negative scores by design

- **Week 1:** Maximin sampling - generalized maximin function selected (0.819854, 0.036171, 0.929157, 0.042194, 0.938540) farthest from existing 20 points for exploration of unexplored 5D regions. **Output:** -2.6736180536592746
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - exploration-focused BO suggested boundary point (0.000000, 0.000000, 0.084701, 1.000000, 0.000000) in 5D space. **Output:** -1.732308297257106 (improvement - less negative)
- **Week 3:** Bayesian Optimization (GP + UCB) - continued exploration, BO exploring boundary regions, suggested (0.230014, 0.000000, 1.000000, 1.000000, 0.000000). **Output:** -1.2341261163607111 (continued improvement)
- **Week 4:** Bayesian Optimization (GP + UCB) - continued exploration-focused approach. **Output:** (BO suggestion from week 4 run)

---

## Function 7 (6D - Maximize)

**Initial Data:** 30 samples, 6 dimensions  
**Description:** ML model hyperparameter tuning - six hyperparameters (learning rate, regularization, etc.)

- **Week 1:** Maximin sampling - generalized maximin function selected (0.057395, 0.931572, 0.975502, 0.005642, 0.900680, 0.173960) farthest from existing 30 points for exploration of unexplored 6D regions. **Output:** 0.013680510023283559
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - exploration-focused BO suggested (0.030175, 0.312148, 0.354869, 0.145624, 0.344577, 0.758289) in 6D space. **Output:** 1.9964705720993239 (significant improvement)
- **Week 3:** Bayesian Optimization (GP + UCB) - BO refining search based on Week 2 success, suggested (0.000000, 0.197220, 0.363340, 0.045543, 0.317971, 0.937275). **Output:** 0.8888266866145418 (lower than Week 2 but still positive)
- **Week 4:** Bayesian Optimization (GP + UCB) - continued exploration, BO learning from accumulated observations. **Output:** (BO suggestion from week 4 run)

---

## Function 8 (8D - Maximize)

**Initial Data:** 40 samples, 8 dimensions  
**Description:** Eight-dimensional black-box optimization (e.g., ML model with 8 hyperparameters)

- **Week 1:** Maximin sampling - generalized maximin function selected (0.824416, 0.925687, 0.201348, 0.929573, 0.017861, 0.924962, 0.951929, 0.012052) farthest from existing 40 points for exploration of unexplored high-dimensional (8D) regions. **Output:** 6.0739207860211
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - exploration-focused BO suggested (0.073701, 0.126448, 0.000000, 0.000000, 0.849435, 0.405574, 0.048621, 0.028310) in 8D space. **Output:** 9.8362097005135 (improvement)
- **Week 3:** Bayesian Optimization (GP + UCB) - continued exploration, BO refining search, suggested (0.025280, 0.201174, 0.025933, 0.196286, 1.000000, 0.666219, 0.190159, 0.771074). **Output:** 9.9008337717904 (slight improvement)
- **Week 4:** Bayesian Optimization (GP + UCB) - continued exploration-focused approach in high-dimensional space. **Output:** (BO suggestion from week 4 run)

---

## Overall Strategy Summary

- **Week 1:** Manual selection for 2D functions, Maximin sampling for higher dimensions - pure exploration selecting points in unexplored regions using basic visualization and distance-based maximin algorithm
- **Week 2:** Bayesian Optimization (GP + UCB, beta=2.0) - moved from manual/heuristic methods to algorithmic BO approach, exploration-focused across all functions
- **Week 3:** Continued Bayesian Optimization (GP + UCB) - continued exploration strategy, identified issues with Function 1 (flat function causing BO to converge to extremes) and manually adjusted
- **Week 4:** Continued Bayesian Optimization, began implementing Logistic Regression for Function 1 - recognized need for function-specific approaches, started logistic regression for Function 1's linear boundary characteristics

---

## Key Observations Across Functions

1. **Function 1:** Shows linear boundary characteristics, BO struggles with flat function, logistic regression identified as better approach
2. **Function 2:** Noisy outputs, some structure but requires careful exploration
3. **Function 3:** All outputs negative so far, likely transformed minimization problem
4. **Function 4:** Dynamic function with local optima, first positive output in Week 2
5. **Function 5:** Appears unimodal with peak in high-dimensional region (~0.9-0.95), BO successfully identified and refining around this region
6. **Function 6:** Negative scores by design, goal is to maximize (bring closer to zero), steady improvement observed
7. **Function 7:** Non-linear hyperparameter space, Week 2 found promising region
8. **Function 8:** High-dimensional (8D), requires extensive exploration, steady progress observed
