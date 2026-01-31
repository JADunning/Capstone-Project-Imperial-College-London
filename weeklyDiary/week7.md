#### Which hyperparameters did you choose to tune, and why did you prioritise them?

For Function 1 (non-linear logistic regression) I tuned: degree (1, 2, 3) to control non-linearity of the decision boundary; threshold_percentile (50, 60, 70, 75, 80, 90) to define “good” vs “bad” outputs for binary labels; C (0.01–100) for inverse regularization strength; **max_iter** (1000, 2000, 5000); and penalty (L2). 

I prioritised degree and threshold_percentile because they most directly affect where the model places the decision boundary and thus which point we query next. And are most easily understood conceptually! I included tuning C and max_iter mainly to avoid underfitting or non-convergence.

For Functions 2–6 the main design choice was β=0 (pure exploitation) in the UCB acquisition. 

For Function 3 I treated the objective as noisy and used a noise-aware Bayesian optimisation surrogate instead of the standard GP. So didn't actually tune anything here.

For Functions 7 and 8 I didn’t actually tune any hyperparameters this week; I used fixed settings for the NN (e.g. hidden=64, epochs=1200, lr=1e-3) and I used gradient-based optimisation (n_restarts=25) to then pick the next point.. I.e. initialise 25 random starting points, perform gradient ascent from each to locate the local maximum of the neural network, and then select the higher predicted point. Maybe next week I can train some hyperparameters! 


#### How has hyperparameter tuning changed your query strategy compared to earlier rounds?
It has meant I'm selecting the next point using a tuned model instead of fixed defaults. So it means I can take the new data, better tune my model, and then aim for a hopefully improved result. 

I've also started doing some progress validation.. i.e. I flag if the new weeks result haven't improved on the maximum. And if they haven't, I go back and re-look at the surrogate. It's why I actually split out a new BO for function 3, because I wasn't getting good results. 


Which tuning method(s) did you apply (manual adjustment, grid search, random search, Bayesian optimisation, Hyperband), and what trade-offs did you notice?
- I used grid search for Function 1’s non-linear logistic regression tuning over:
degree -> 3 values
threshold_percentile: 6 values → 50, 60, 70, 75, 80, 90
penalty: 1 value → 'l2'
C: 5 values → 0.01, 0.1, 1.0, 10.0, 100.0
max_iter: 3 values → 1000, 2000, 5000

So 3 x 6 x 1 x 5 x 3 = 270 combinations

So it was relatively easy to compute and run through them all. And then evaluated by MSE. So also quite easy to interpret. But I did constrain the hyperparameters so it wouldn't take long, there was a choice in the bands I used for those. If I had more it would have taken too long.

I did manual/design choices for BO: β=0, progress-validation threshold, noise-aware BO for Function 3, and n_restarts for gradient optimisation. Just because it kept it simple! Although the trade off here is that I might actually not be finding an optimum. 

I did not use Hyperband or Bayesian hyperparameter optimisation in week 6.


#### As your data set grows to 16 points, what limitations of your model become clearer through tuning (e.g. overfitting, irrelevant features, diminishing returns)?
For function 3, the noise in the objective becomes clearer - outputs do not improve smoothly across rounds, and so maybe the standard GP I fit was mislead a bit by the noise. 

In the polynomial linear regression, since I have only 16 points, overfitting is a real concern and risk, especially since my best configs achieved very low MSE on the small training set. 

You also get diminising returns with these small datasets when tuning, i.e. many C and max_iter values gave the same performance or didn't really change much. I probably should not bother tuning them next round!


### How might you apply hyperparameter tuning techniques to larger data sets in future rounds of the BBO capstone project submissions or more complex models in future ML/AI projects?

For larger data sets I could replace a full grid search with a random search, let's see if my computer can handle the increase processing needed. 

I could think about using Bayesien Optimisation to tune my hyperparamters in the neural network. I could start tuning the hyperparameters in my neural network with more data ! Since there would be less risk of overfitting. 



#### How does tuning in this black-box set-up prepare you to think like a professional ML/AI practitioner in real-world contexts with incomplete information?
Yeah it's great! In a real world scenario you often have limited data and limited ability to do model runs. So this trains you to do things like:
1. Choosing what to tune based on impact and cost.
2. Rely more on validation, i.e. looking at the results and making sure you understand them. Which again means you think more about the tuning, what methods had a good result / benefit. For example if you're plateauing, maybe you should try something else instead of keeping on tuning.

