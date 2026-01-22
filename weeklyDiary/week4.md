#### In your function evaluations, which inputs seemed to act like support vectors – points near a decision boundary or region of rapid change? How might recognising them guide your next query?

Function 1 had some inputs that seemed to act like support vectors, I think because this function has data that sits in 2 categories. It seems to fit a logistic regression quite well!

What I did this time was fit and trained a simple linear regression for this function and I have chosen a point that sits right on the decision boundary. Depending on the output I will the know if my logistic regression model is correct or not. 

I have a feeling, looking at the graphs. I have chosen a linear logistic regression, but maybe it needs to be non-linear with a curved decision boundary.



#### If you trained a neural network or another surrogate model, did you explore how the outputs change in response to the inputs? How might these gradients point to directions that reduce the function value? If you did not train a neural network or surrogate model, explain why you chose not to. 

I haven't yet chosen to train a neural network yet because:
- Some functions don't suit a neural network. I.e. function 1 it seemed like a better decision to train a logistic regression. Other low dimension functions might not fit a neural network which is designed for complex, non-linear relationships.
- We have a small amount of data points (13) so we risk overfitting at this point, for many of the functions bayesian optimisation is likely still a better choice.

I will look at training a simple neural network for some of the higher dimension functions slightly later on. 

If I was using a neural network, the gradient points in the direction of steepest increase, so the negative gradient points in the direction of steepest decrease - by moving opposite to the gradient, you follow the downhill direction to reduce the function value.


#### Imagine framing your BBO capstone project as a classification task (‘good’ vs ‘bad’ outputs). How could models such as logistic regression, SVMs or neural networks capture this decision boundary? What trade-offs would you face between misclassification and exploration?

For Function 1, I framed BBO as a classification task by converting continuous outputs into binary classes using a threshold (75th percentile), labeling outputs above the threshold as "good" (Class 1) and below as "bad" (Class 0). I chose logistic regression to learn a linear decision boundary, which is appropriate for 13 data points and a 2D input space. Logistic regression outputs probabilities at any point, allowing me to identify the boundary (where P(Class 1) = 0.5); SVMs could provide similar results with linear kernels, while neural networks would overfit with so little data. The trade-off is accepting some misclassification near the boundary to prioritize exploration — I query points where P(Class 1) ≈ 0.5 (highest uncertainty) rather than maximizing training accuracy, which guides active learning toward refining the boundary and discovering high-output regions.


#### Which type of model – linear regression, SVM or neural network – felt most appropriate for guiding your search? How did you balance interpretability against flexibility when making this choice?

For Function 1, I chose logistic regression over linear regression, SVMs, and neural networks. With 13 data points and a 2D input space, I prioritized interpretability and generalization over flexibility. Logistic regression provides clear probability estimates (P(Class 1)) that directly guide exploration, whereas SVMs and neural networks would likely overfit with so little data. While a non-linear model could capture more complex boundaries, a simple linear decision boundary is sufficient for 2D optimization and avoids overfitting. This choice balances interpretability (clear probability outputs for active learning) with sufficient flexibility (linear boundary appropriate for low-dimensional space), making it more effective for guiding search with limited data than more complex alternatives.



#### Looking at your neural network surrogate, which input variables showed the steepest gradients or the greatest influence on your predictions? How might you use this to prioritise your next experiments?

I haven't coded a neural network surrogate.



#### When framing your BBO problem as a classification task (‘good’ vs ‘bad’ outputs), how effectively did your neural network approximate the decision boundary? In what ways did backpropagation help you interpret or visualise this boundary?

I used logistic regression instead of a neural network because, with only 13 data points, a neural network would likely overfit. Logistic regression provided a clear linear decision boundary appropriate for the 2D input space and the available data. While backpropagation would help interpret weights and gradients in a neural network, logistic regression directly outputs interpretable probabilities at any point, which guided my active learning strategy. The linear decision boundary (where P(Class 1) = 0.5) effectively separated high and low output regions, and I visualized it using probability contours and uncertainty maps to identify the next query point near the boundary.



#### Compared to simpler models such as linear or logistic regression, how well did your neural network capture non-linear patterns in the function? Was the added flexibility worth the extra complexity in tuning and interpretation?

I used logistic regression rather than a neural network, as the 2D input space and 13 data points make a linear decision boundary sufficient. The model captured the main pattern separating high and low output regions. A neural network’s added flexibility (non-linear boundaries) would not have been worth it: it would likely overfit with so few points, require more hyperparameter tuning, and reduce interpretability (no clear probabilities). The linear boundary from logistic regression balances accuracy and interpretability for this low-dimensional optimization problem, so the extra complexity would not have been justified.





TASK 2
Hyperparameter effects
I tuned hyperparameters for logistic regression (not a neural network) due to limited data. Key hyperparameters:
- threshold_percentile (50, 60, 70, 75, 80, 90): Controls class balance. Lower values increased Class 1 samples but reduced separation; higher values improved separation but reduced Class 1 samples. The best model used 50th percentile, balancing both classes.

- C (0.01, 0.1, 1.0, 10.0, 100.0): Inverse regularization strength. Higher C reduced regularization, increasing overfitting risk; lower C increased regularization, improving generalization. The best model used C=10.0.

- max_iter (1000, 2000, 5000): Maximum iterations. Higher values improved convergence but increased training time. 2000 iterations was sufficient.

- penalty (None, 'l2'): Regularization type. L2 helped prevent overfitting; the best model used L2 with C=10.0.

For neural networks, key hyperparameters include learning rate, batch size, number of layers/neurons, activation functions, and dropout. I did not use a neural network because with 13 data points it would likely overfit.

Discrete vs continuous
Discrete hyperparameters:
threshold_percentile: 50, 60, 70, 75, 80, 90 (categorical/ordinal)
penalty: None, 'l2' (categorical)
max_iter: 1000, 2000, 5000 (integer)

Continuous hyperparameters:
C: 0.01 to 100.0 (continuous, log scale)

Tuning method influence:
Discrete: Grid search or random search over fixed values
Continuous: Grid search with chosen values, random search, or Bayesian optimization over ranges
I used grid search across all combinations, which works for both types but scales poorly with more hyperparameters.

Application to the capstone
For Function 1 (2D, 13 points), I used logistic regression with systematic hyperparameter tuning across 180 combinations, finding that L2 regularization with C=10.0 best balanced accuracy and generalization. For higher-dimensional functions (3–8D), I plan to use neural networks, and I will apply Bayesian optimization to tune hyperparameters like learning rate, batch size, architecture, and dropout, treating hyperparameter selection as an optimization problem to efficiently search large spaces. The systematic approach from Function 1—evaluating multiple configurations and selecting based on validation performance—will guide this process, using BBO to more efficiently identify effective neural network configurations than grid search, especially for higher-dimensional problems with limited data.