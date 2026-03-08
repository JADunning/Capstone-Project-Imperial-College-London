#### How has your optimisation strategy evolved since your first few rounds of queries? Which elements now feel more structured or systematic?

I have certainly moved from exploration to exploitation. 

The first few weeks I applied a very similar exploration based BO to all functions. Then after a few weeks, I moved to exploitation based approaches, which was aiming to capitalise on the promising areas. When I did this my strategy per functions started to split, for example, I quickly created a logistic regression function for function 1. A few weeks after that I built a simple neural network for function 7 and 8. A few weeks after that I used a more noise suitable BO for function 3 and eventually function 5.

And it wasn't just the functions I used that evolved throughout the weeks, but the methods I used to inspect the data. 
At the start I was plotting simple graphs, often just using the first 2 dimentions. I quickly moved to t-SNE to better visualise multiple dimentions. I also started introducing results validation later in the weeks, originally just analysing the results form the last week to see if it had improved or not, but then eventually analysing the output from the models I was choosing, and seeing how the distance of that point compared to the current highest. This meant I was always searched in a more structured way around the current optimum.




#### If you think of your current data set as a ‘high-dimensional’ space, which variables or behaviours seem to drive the largest variation in your results – similar to principal components in PCA?

I think about this in two ways.

1. Within each function (input dimensions) 
The "space" is the input cube for that function. The "result" is the scalar output. What drives the largest variation in the output is effectively which input dimensions (or directions) matter most—like the first principal components of the inputs when you view the output as the quantity to explain. For Function 1 I get this from the logistic regression: the coefficients (and polynomial terms) show which directions in (x1, x2) separate high vs low output, i.e. which dimensions have the strongest influence. For higher-dimensional functions I don't have an explicit PCA on (inputs → output), but the same idea appears in the GP length scales (which dimensions the surrogate thinks are most important) and in t-SNE: the 2D embedding preserves structure so that directions of large variation in the data often align with where the output changes most. So in practice, the "first principal component" per function is roughly: the direction(s) in input space along which the output varies most, which I infer from the surrogate (logistic boundary, GP, or NN) and from visual structure in t-SNE.

2. Across the project (meta-dimensions)
The "dataset" here is all runs: eight functions × many weeks × different strategies. The "results" are things like best output so far, week-on-week improvement, and distance to incumbent. The dimensions of this meta-space include: function identity (each function has different scale, noise, and geometry), input dimensionality (2 vs 3 vs … vs 8), noise (Function 3 vs others), strategy (logistic vs GP vs NN), and time/phase (early exploration vs later exploitation). The "principal components" that seem to explain most of the variation in outcomes are: (1) which function—performance and difficulty vary a lot by function; (2) dimensionality—drives method choice and how hard optimisation is; (3) noise—Function 3 forces a different strategy (noise-aware BO) and more variable results; (4) strategy/match—using the right method for the function (e.g. logistic for 2D, NN for 6–8D) drives a lot of the variation in success. So in a PCA-like sense, the first "component" is function identity and problem type, and the next is the strategy–problem match (method vs dimensionality and noise), with redundancy reduced by not using one universal approach but specialising per function.

#### How do you decide which aspects of your strategy to keep exploring versus which to reduce or simplify, as PCA reduces dimensions while retaining essential information?

When deciding on the meta-dimentions, i.e. which function to focus on / improve this week. 

I did this by 1) visualising all the results from the previous week. 2) Making use of validation functions which show me if the result is improving or not, 3) Using a table for all the results showing what the maximum was, and how close my output is to that maximum. 

The idea being, any functions that are stalling in progress or missing the mark, I can personally see and identify these, and then do some work on that function this week to make sure we get an improvement. 

In terms of improving the input dimensions, I decide what to keep or simplify by leaning on the surrogate: coefficients (logistic) or length scales (GP) show which dimensions drive the output, so I focus there and treat the rest as secondary—like PCA keeping high-variance components. I also use MSE: low MSE means I trust and exploit the surrogate; high MSE means explore more or simplify the model.


#### How might this round of optimisation influence your next and final round of query submission in Module 24, especially when balancing exploration and exploitation?

We only have 1 round left! So there isn't enough time to do any explanation, I'm probably just going to do a final round of exploitation and see if I can further improve on my best results. 

#### Reflect briefly on how insights from PCA, such as focusing on variance and removing redundancy, might apply to how you interpret your BBO results.

Like PCA, I focus on what explains the most variation (which dimensions or functions drive outcomes) and treat the rest as secondary. I reduce redundancy by using one strategy per function instead of a single universal approach, and by letting coefficients or length scales tell me where to concentrate—so I keep the “essential” structure and ignore noise or less informative dimensions when deciding where to query next.

