Respond to the following prompts:
### CNNs build up features from edges and textures to full objects. How did this idea of progressive feature extraction influence the way you thought about refining your BBO strategy?

The idea of progressive feature extraction influenced my BBO strategy by encouraging a layered approach. Just as CNNs build from edges → textures → objects, I progressed (in functions 2 to 6 where I'm sticking with BO) from broad exploration → exploitative optimisation → automatic progress validation. Similarly (with function 1), I moved from linear to polynomial logistic regression, building complexity incrementally in these functions. And for functions 7 and 8 from explorative BO -> last week using a neural network with random sampling to now adding gradient-based optimisation to find the optima. This simple -> complex iteration mirrors how CNNs refine from simple pattern detection to sophisticated feature combination.


### LeNet and later CNNs redefined what is possible in computer vision. What parallels do you see between those breakthroughs and the incremental improvements you make in your BBO capstone project?

LeNet and later CNNs showed how incremental improvements compound into major breakthroughs. Similarly, my BBO improvements accumulate: each week builds on prior insights—from linear to polynomial regression, from random sampling to gradient optimisation, from basic exploitation to validated progress tracking. Just as LeNet proved deep learning could work, then AlexNet showed it could scale, my incremental refinements compound into better performance.


### Training CNNs often involves balancing depth, computational costs and overfitting risks. Did you face similar trade-offs when choosing whether to explore widely or exploit promising regions in your queries?
Yes I suppose - similar trade-offs. Just as deeper CNNs improve performance but increase computational cost and overfitting risk, I balance exploration (querying uncertain regions) against exploitation (refining known promising areas). Early iterations explore broadly to map the function landscape; later iterations exploit, similar to fine-tuning. This week's progress validation acts like early stopping—it flags when exploitation plateaus, preventing overfitting to a local region. 

While I can run everything locally without heavy computational constraints, I still face efficiency trade-offs: random sampling (50,000 evaluations) is thorough but slower, while gradient optimisation (~2,500 evaluations) is more efficient and finds better optima—showing how better techniques improve performance even when computational resources aren't the primary constraint.

### Convolutions, pooling, activations and loss functions influence how CNNs learn from data. Which of these concepts helped you think differently about how your optimisation model learns from your accumulated data?
Loss functions and gradients helped most. Using a threshold (75th percentile) to define "good" vs "bad" outputs acts like a loss function that guides what the model learns. Probabilities (activations) from logistic regression guide query selection—I query points where P(Class 1) ≈ 0.5 (highest uncertainty), similar to how activations indicate model confidence. This week, I directly use gradients: for functions 7-8, gradient-based optimization uses the neural network's gradients to climb to optima, similar to how CNNs use backpropagation. Progress validation acts like a loss function that flags when learning plateaus, guiding when to change strategy. This uncertainty - and gradient-based approach helps the model learn efficiently by focusing on the most informative regions.


### The interview with Andrea Dunbar highlighted the trade-offs of deploying CNNs in edge AI systems. How might reflecting on real-world deployment challenges help you decide how to benchmark success in your own BBO capstone project?

Edge AI deployment challenges shifted my benchmark from peak performance to consistency. Just as edge AI systems must work reliably under constraints, I should evaluate my BBO methods by consistent improvement across rounds given limited queries, not just maximum function values.




### What is the main technical justification for your current BBO approach? Which aspect of prior research or established methods supports your choice?
My approach is justified by established Bayesian Optimisation principles. I use Gaussian Process regression as the surrogate model (standard in BO) with Upper Confidence Bound (UCB) acquisition. For functions 2-6, I use exploitative BO (β=0) to refine promising regions after initial exploration—this follows the exploration-exploitation trade-off. For functions 7-8, I use neural network surrogates with gradient-based optimisation, which is more efficient in high dimensions where GP becomes expensive. The method selection by function complexity (logistic regression for 2D, GP for 2-6D, NN for 6-8D) follows the principle of matching method complexity to problem dimensionality.


### Which academic papers have you used to guide your design? Which ideas or techniques from the literature are most relevant, and how do they strengthen your project?
My approach is guided by well-established ideas from the Bayesian Optimisation and active learning literature. In particular, classic work on Gaussian Process–based Bayesian Optimisation, such as Jones et al.’s Efficient Global Optimization of Expensive Black-Box Functions (1998), introduced the use of surrogate models and acquisition functions (e.g. Expected Improvement) to efficiently optimise expensive black-box functions. This work motivates the general surrogate-based optimisation framework used in this project.

The use of an Upper Confidence Bound (UCB-style) acquisition reflects the broader exploration–exploitation trade-off studied in later Bayesian optimisation and bandit literature.


### Which third-party libraries or frameworks (e.g. PyTorch, TensorFlow, scikit-learn) are central to your approach? Why were these the right choices compared with possible alternatives?
The central libraries are scikit-learn, scipy, PyTorch, and numpy. scikit-learn provides Gaussian Process regression, logistic regression, and polynomial features with minimal code—crucial for rapid iteration with limited data (13-16 points per function). scipy offers reliable optimization routines (L-BFGS, differential evolution) for optimizing acquisition functions. PyTorch enables rapid experimentation with neural network surrogates and gradient-based optimization. numpy handles data manipulation. These choices fit black-box optimization with limited queries: they're well-tested, integrate well, and enable quick prototyping. The trade-off is less control over hyperparameters compared to custom implementations, but the convenience and reliability outweigh this for this project's scale.


### How do you plan to document and present these justifications in your GitHub repository so that peers, facilitators and future employers can clearly understand your reasoning?
I'll document justifications in the README and in notebook comments. The README will explain the technical approach, method selection rationale, and library choices. Each notebook includes comments explaining why specific methods are used for each function type. The results/ directory stores outputs with metadata about methods and hyperparameters. I'll add a "METHODOLOGY.md" file explaining the progression from exploration → exploitation → validation, and why different methods suit different function complexities. This makes the reasoning clear to peers, facilitators, and future employers.


### Looking ahead, what additional sources (research, benchmarks, software) might you consult to continue refining your strategy?
I'd consult BO literature, benchmarks, and specialized libraries. Key sources: foundational BO papers (Mockus, Jones et al.), recent work on high-dimensional BO and neural network surrogates, benchmark suites (e.g., COCO for continuous optimization), and specialized BO libraries (e.g., BoTorch, GPyOpt) to compare approaches. I'd also look at active learning literature for uncertainty-based query selection and multi-fidelity optimization for handling limited queries. These would help refine the strategy, validate against benchmarks, and explore advanced techniques like ensemble methods or adaptive acquisition functions.