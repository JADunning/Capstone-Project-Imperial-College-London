#### How did the ideas of hierarchical feature learning influence the way you thought about structuring or refining your optimisation strategy this round?

Hierarchical feature learning is how deep neural networks learn in layers, with each layer building on the previous one. E.g. 
1. Lower layers learn simple, local patterns (edges, textures etc.)
2. Middle layers combine these into more complex patterns
3. Higher layers then learn and abstract high-level concepts.

In my optimisation strategy this round, I could apply a similar layers or hierachical approach. 
- A good example is with my logistic regression, I started last week with a simple linear logistic regression. Building on the learning from that, i have now added in a more complex non-linear regression.
- Or with my Bayesian Optimisation, early iterations explored broadly, now i'm moving more to exploitation and refining. This is similar to how networks first learn general features and then specialise.

So just like neaural networks learn, I'm adapting my strategy by building understanding of each function in stages -> often from coarse to finer / more complex features.



#### You saw how breakthroughs such as AlexNet and ImageNet classification reshaped expectations in AI. What parallels do you see between those leaps in performance and the incremental improvements you make in your capstone submissions?

AlexNet and ImageNet showed how incremental improvements compound into major gains. Similarly, I progressed from linear to non-linear logistic regression and from broad exploration to targeted exploitation in Bayesian optimization, with each step building on prior insights. Just as those breakthroughs came from validated iterations rather than single leaps, my improvements accumulate—better surrogate models lead to better query selection, which improves overall optimization performance.



#### When training neural networks, people often weigh trade-offs between depth, complexity and training efficiency. Did you encounter similar trade-offs in deciding whether to explore widely or exploit known promising regions in your queries?

Yes—similar trade-offs. Just as deeper networks improve performance but cost training time, I balance exploration (querying uncertain regions) against exploitation (refining known promising areas). Early iterations explore broadly to map the function landscape; later iterations exploit, similar to fine-tuning a trained network. 



#### Reflecting on the building blocks of neural networks (inputs, activations, loss, gradients, weight updates), which of these concepts helped you think differently about how your model learns from the data you’ve accumulated so far?

The loss/threshold concept helped most: using a threshold (75th percentile) to define "good" vs "bad" outputs is like a loss function that guides what the model learns. Probabilities (activations) from logistic regression guide query selection—I query points where P(Class 1) ≈ 0.5 (highest uncertainty), similar to how activations indicate model confidence. This uncertainty-based approach, analogous to gradient magnitude indicating where the model is most uncertain, helps the model learn efficiently by focusing on the most informative regions rather than just maximizing accuracy.


#### Module 16 also introduced PyTorch and TensorFlow as different frameworks for building and scaling models. If you were to frame your current optimisation approach in terms of a ‘framework’, would it be closer to rapid prototyping and flexibility or to structured, production-ready design? Why?

My approach is closer to rapid prototyping and flexibility, similar to PyTorch. I iterate quickly—switching from linear to non-linear logistic regression, adjusting polynomial degrees, and tuning hyperparameters—which requires flexible, easy-to-modify code rather than production-hardened infrastructure. The focus is on experimentation and learning what works for each function, not deployment or scaling, so I prioritise quick iteration over structured, production-ready design. 



#### In the guest interview, Giovanni Liotta discussed industry applications of deep learning in sport. How might reflecting on real-world deep learning use cases inform the way you benchmark success in your own capstone challenge?

In sports analytics, success is measured by real-world impact—whether insights improve team performance or inform decisions—not just model accuracy. This shifts my benchmark from maximizing function outputs to understanding why certain regions perform well - I should really look back over the description of what each function is more! And relate my success to that.




Reflect on the following prompts, and post your responses on the discussion board.

Repository structure
#### How have you organised your repository so far (e.g. data, notebooks, queries, results)?

My repository is organised into several key directories: data/ contains initial data organized by function (function_1 through function_8) and weekly query results (week1/, week2/, etc.) with inputs.txt and outputs.txt files. 

weeklyWork/ contains analysis notebooks organized by week (e.g., week4/function1_logistic_regression.ipynb, week5/function1_nonlinear_logistic_regression.ipynb), while weeklyDiary/ stores weekly reflection documents. 

Python modules live under scripts/ (e.g. scripts/bayesian_optimisation/, scripts/logistic_regression/, scripts/utils/data_utils.py) for organised imports.

#### What changes will you make to improve clarity, navigability and reproducibility?

I plan to: (1) consolidate all notebooks into weeklyWork/ to avoid clutter, (2) add a results/ directory to store best points found, performance metrics, and visualizations, (3) create a scripts/ directory for reusable utility functions, (4) add docstrings and comments to improve code readability, (5) include a QUICKSTART.md with step-by-step instructions for running the latest strategy, and (6) update the README's project structure section to reflect the current organisation.

Coding libraries and packages
#### Which libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach?

The central libraries are 
- scikit-learn (for logistic regression, polynomial features, and Gaussian processes), 
- scipy (for optimisation routines that help find the best points—I use differential_evolution to search globally for points near decision boundaries), and 
- numpy/pandas (for data manipulation). 
- Pytorch is used for a simple neural network


#### Why are these choices appropriate for your problem, and what trade-offs did you consider?

These choices fit the constraints of black-box optimisation with limited queries. scikit-learn provides logistic regression, polynomial features, and Gaussian processes with minimal code, enabling quick iteration—important when testing strategies with only 13–16 data points per function. The trade-off is less control over hyperparameters compared to custom implementations, but the convenience outweighs this for rapid prototyping.

scipy offers reliable optimisation routines that are well-tested and handle the global search needed to find points near decision boundaries and optimise acquisition functions. The trade-off is that these are general-purpose tools rather than specialised for BBO, but they're sufficient for this project's scale.

numpy/pandas are standard for data manipulation and integrate well with scikit-learn, making data loading and preprocessing straightforward.

PyTorch fits this project because it supports rapid experimentation with surrogate models. This research-oriented design aligns with my experimental approach, where understanding what works matters more than production deployment.


Documentation
#### How do your README and other documents currently describe the purpose, inputs, outputs and objectives of your BBO capstone project?

README is organised into 5 main sections: Project Overview (what BBO is, real-world relevance), Inputs and Outputs (formats with examples), Challenge Objectives (primary goal and constraints), Technical Approach (10-week strategy), and Setup Instructions (environment and dependencies).

#### What updates do you need to align the documentation with your most recent strategy and results?
I need to update the technical approach section to reflect what i've actually done! And update the project structure to include the recent weeks and new modules. And then finally, I should really add a results section!