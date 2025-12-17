#### How has your query strategy changed from earlier rounds? Do you rely more on model predictions, or are you still exploring new regions? Do you tune hyperparameters or rely on heuristics?

My strategy hasn't changed at all, it's week 3 so I'm sticking with exploration focused Bayesien Optimisation. This is part of the original strategy, to focus on exploration for a few weeks before then exploring other alternatives and maybe some apecific models / techniques for specific algorithms.


#### How do you balance exploration against exploitation? Do you focus more on areas known to perform well, or are you still sampling from untested regions?

Still sampling from untested regions since we're still early on in the process!


#### How would SVMs change your approach? Could you use a soft-margin SVM to classify high vs low performance regions? Would a kernel SVM help if the response surface is non-linear?

SVMs are used in classification problems, so they can't be directly applied to this problem. But I could use it to split the space into high performance vs low performance as suggested. This could be an interesting approach to then narrow the exploration space further as we transition from exploration to exploitation. The SVM trick in this black box exercise would likely struggle with the lack of data.

A kernal SVM would help if the response surface is non-linear since it implicitly maps the data into a higher dimensional feature space where a linear decision boundary becomes possible. 


#### What limitations of your current model become apparent as data grows? Is it overfitting? Do any features or dimensions emerge as irrelevant?

I am struggling with the first function! The model is struggling with this function and has given me an output the same as last week (0,0). 

Because the function is so flat, and everything looks the same, it's tending to extremes. 

I could fix this by adjusting the Bayesien Optimisation and forcing exploration. But instead, I really should move to logistic regression.

For now, I'm going to pick a point close to the transition which is 0.6,0.6.. I will code a logistic regression next week! 


#### How does this black-box set-up prepare you to think like a data scientist when faced with incomplete knowledge in other projects?

In other projects you often have limited data, limited compute, incomplete knowledge. So you have to spent time trialling techniques, understanding the data, and really strategising instead of just throwing compute at everything.
