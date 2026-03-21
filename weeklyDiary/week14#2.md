#### Which optimisation strategies led to your strongest results, and why were they effective? How did these strategies influence your decisions as the challenge progressed?

My strongest results came from strategies that matched the structure of each function rather than forcing one method onto everything. For Function 1, non-linear logistic regression worked better than shared BO because it captured the narrow "good" region more clearly. For Functions 2, 4, 5, and 6, Bayesian optimisation was strongest once I moved from early exploration to more exploitative and then more robust local trust-region variants. For Functions 7 and 8, the neural-network surrogate plus gradient optimisation worked well because it handled the higher dimensionality better than a simple GP. As the challenge progressed, this pushed me toward a more function-specific strategy and away from a one-size-fits-all workflow.

#### In your view, what qualities define a 'successful' strategy - is it outcomes alone or also adaptability, reasoning or efficiency?

I think a successful strategy is not just about the final output. Outcomes matter, but adaptability, reasoning, and efficiency matter too. In this project, a strategy was only really successful if I could explain why I was using it, tell whether it was improving, and adjust it when it stopped working. A method that gives one strong result by luck is less useful than a method that is understandable, repeatable, and robust under a limited query budget.

#### How could the strategies you identified be applied or adapted to professional ML/AI projects beyond the BBO capstone project?

These strategies transfer well to professional ML work because real projects also require limited-budget decision making, model selection, and structured iteration. The main lesson is to match the method to the problem: use simpler, interpretable methods when they fit; use more flexible models when dimensionality or complexity requires it; and always build validation into the workflow. The same thinking applies in areas like hyperparameter tuning, experimental design, A/B testing, and any ML setting where you need to learn efficiently from limited expensive observations.

#### What successful strategies did you notice in your peers' approaches, and what made them effective? Do you see overlap with your own strategy? Explain your reasoning.

One successful pattern I noticed in peers' work was the same general idea of incremental refinement: start with a simple method, learn from the results, then increase complexity only when needed. For example, the strategy of moving from simpler surrogate models toward trust-region or GP-based optimisation mirrors my own shift from broad BO to more specialised and robust pipelines. I also saw overlap in the balance between exploration and exploitation, and in using structured weekly feedback to decide whether to keep refining a region or rethink the method.

#### What suggestions or perspectives could strengthen your peers' strategies, and how do their reflections broaden your view of what success means in optimisation?

One suggestion that could strengthen many strategies, including my own, is to introduce validation checks earlier so it is easier to spot when the optimiser has drifted away from a promising region. I also think it helps to define clear switching rules for when to move from exploration to exploitation or from one surrogate model to another. Reading peers' reflections broadened my view of success because it showed that optimisation is not only about who got the highest score. It is also about whether the approach was well reasoned, whether the trade-offs were understood, and whether the workflow became more disciplined and reproducible over time.
