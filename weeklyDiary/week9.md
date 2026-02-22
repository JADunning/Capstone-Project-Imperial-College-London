#### How do scaling laws influence your current query choices? Do you see diminishing returns or steady improvements?
Scaling laws influence current query choices:
1. In function 5 I am seeing a good example of diminishing returns. Each new submission now tends to add only small gains compared to earlier, more exploratory steps. You’re seeing clear diminishing returns where we get much smaller improvements as you approach the maximum. 
2. Another example of scaling laws influencing my decisions might be in function 7 and 8 where I am using a simple neural network. My current set up uses relatively little compute. Scaling laws suggest that with more training epochs and a larger network I could improve the surrogate and thus the accuracy of the next point suggestions. But I'm currently not pushing compute hard, because I'm balancing this relatively small increase in accuracy against run time and practical constraints of doing this on my own laptop.


#### Where might emergent behaviours alter your expectations, and how are you preparing for them?
Emergent behavious would be sudden or unexpected changes in our function results of function behaviour.

This could be like in function 3 where there's a lot of noise, which often comes with unexpected results. To prepare for this I am using a more noise-aware Bayesian optimisation and re-running the same input multiple times to see if I can distinguish noise from real changes. 

Function 1 could be another example where you see an emergence behaviour, because it has been mostly zeros, with one then spike into a positive number, and mostly zeros since. I actually need to change my tactic here for the last few weeks, and focus more on points around that high number to see if I can better find the decision boundary.


#### What trade-offs between cost, robustness and performance are shaping your strategy now?
I’m trading off compute and complexity against accuracy for the most part. 

E.g. For functions 7 and 8, I use modest NN settings (e.g. 1200 epochs, 64 hidden units, 25 restarts) instead of pushing compute, so runtime stays manageable on my laptop. I accept some loss in surrogate accuracy for lower cost. 

However I do favour robustness where it matters, a good example is the noise-aware BO for function 3, or the progress validation I have built to help me focus on functions that are stuck in terms of progress. Because of the limited (personal) time in to work on this project, focusing on the highest impact thing.. which is what this allows me to do, should give the biggest jumps in accuracy overall. 


#### How do you balance predictable optimisation with the risk of sudden but uneven emergent capabilities?

I lean on predictable methods in my approach e.g. exploitative BO (β=0) and progress validation so improvements are steady and interpretable. 

I also favour checking when a result looks unusual, e.g. noisy function 3 and 5, I am checking the local optima again - again leaning more towards predictable and interpretable optimisation.

And I'm generally keeping things simple where possible (less hyperparameters, or using modest compute) just so it's easy to notice when behaviour doesn’t match expectations and to adjust.

This is a learning project, not just pure performance competition, so I'm leaning more on predictable methods because they support understanding and iteration.

Given this lean towards a more inerpretable / less black box approach, I might not see many sudden or uneven emergent capabilities. (unless the data is noisy)


