#### What was the main change in your strategy in this week compared to the last week? What prompted this change? Was it the model predictions, acquisition function behaviour or something else?

This week I used a simple Bayesien Optimisation model fit for ALL functions
(GP + UCB). It has a simple Gaussian Process surrogate and asimple acquisition function.

Use the same BO for all functions. 

Because I was still focusing on exploration this week. But I wanted to move to a BO process and not stick with manual interpretation.



#### Did you focus more on exploration (sampling uncertain areas) or exploitation (targeting promising areas)? Why? What trade-offs did you weigh?

I focused more on exploration, because we're only in week 2 and we only have 11 data points so far. So for all functions I think it's better to keep exploring for now.


#### Have any participant strategies, class discussions or recent outputs influenced how you approached this week's submission?

The Bayesien Optimisation classes influenced my approach, especially the discussion around when to switch between exploration and exploitation. Since we have 10 weeks and 10 submissions, it makes sense to use at least 3 on exploration before trying to exploit anything.



#### If you were to fit a simple linear or logistic regression model to your current data for one of the functions, which assumptions would you most likely violate? Consider aspects such as the shape of the response surface, noise levels or number of features.

Linear regression assumes the function is a flat hyperplane. In reality all of these black box functions are highly non-linear, possibly with multiple peaks, and have interactions between the differing variables. All of these points would violate that linear assumption



#### Are there any regions where the output appears roughly linear or where a decision boundary might form? How might a logistic regression classifier perform on this function, particularly in binary or threshold-based scenarios?

The first function does seem to have quite a linear boundary, since most of the outputs tend towards 0. Here a logistic regression classifier might peform well on this function

For Function 1 (2D), a logistic surrogate could:
- Approximately identify the region where the function is increasing
- Act as a directional heuristic (“move that way for better scores”)
- Provide a probability map over the space
- Help choose the next point to explore


#### Interpretability is a key advantage of linear and logistic regression. Did you find it useful to consider individual feature effects before deciding on your query point?
Yes - the interpretability of linear/logistic regression could be useful, but only in a limited way. By fitting simple models to the small dataset, I could quickly see which input dimensions appeared to have the strongest influence on the output, and in what direction. This would help me understand the “shape” of the function locally and gave a rough sense of which direction in the input space was more promising.

However, the effects were not globally reliable because the underlying functions are likely nonlinear and high dimensional. So you could use these functions as intuition aids, rather than definitive guides. 

They also are likely more helpful in lower-dimensional functions (especially function 1 where the data does suggest there is a roughly linear boundary)

