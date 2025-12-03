### IDEAS ON WHAT TO DO NEXT

1. BETTER VISUALISATION:
Use t-SNE to visualise data - takes high dimensional data, maps them to 2D / 3D and tries to preserve local neighbourhoods 

2. PLAN OF ATTACK

WEEK 1
- Simple visualisation + understanding of the data

WEEK 2
- Get a simple Bayesien Optimisation model fit for ALL functions
(GP + UCB)

It shoud have a simple Gaussian Process surrogate

A simple acquisition function

Use the same BO for all functions

- Sanity check the outputs visually.


WEEK 3
- Improve the surrogate model robustness by tuning hyperparameters.
- So fit multiple models on the same data we already have -> compare these options without querying the function
- Then pick the one configuration you think best fits.



WEEK 8




### QUESTIONS

What to do week on week -> could try different surrogate models - e.g. random foreast / linear model
-> Could try a mix of exploration vs exploitation. 
FIRST 3 WEEKS go with exploration. Explore data which are very far away. 
From week 4 then look at other inputs around it.

I suppose without knowing the right answer and only with 10, we're spending a lot of time just trying to work with the data we have..



1. SPECIFICALLLY WITH THE FIRST FUNCTION, HOW DO YOU OPTIMISE SOMETHING THAT IS MOSTLY 0?






PLAN
- Some people move into neural netwrok for 6,7,8 because it can perform better for the multi-dimensaional ones
- Function 3 has a lot of noise. Therefore error is not the best.
- FUNCTION 1 when can try logistic regression instead of GP.

Then keep Bayesian Optimisation for everything else. After week 8 you could try decision tree for

(try gradient descendents <- look into this if you have time)


How to score the function -> intensity of the error that is generated. MSE is more than enough. Function 7 and 8 maybe RMSE is better (if MSE is difficult to interpret)




Try to only spend 1 or 2 hours per week on the competition. This is where the core learning is coming from. 1/2 hours per week. 