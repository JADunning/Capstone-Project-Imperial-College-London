#### what reasoning guided your submission for this tenth round? Explain your strategy. How did patterns from the previous rounds influence your decisions for each function?

I made a table this week which compared:
- Output
- Euclidean distance of input to maximum output
- Progress summary (whether we improved on last week or not)

I wanted to do some quick analysis to show we were making progress and optimising around the current known maximum. Where we weren't I then spent some time on those functions, and made sure we optimised further. 

The main thing I found, I had some BO where they weren't optimising around the maxima. For these functions I reworked the BO to use the Expected Improvement (EI) acquisition script, which balances exploration and exploitation and tends to refine around the current best region instead of drifting or getting stuck like the mean-only (exploitative) variant.

#### How transparent is your decision-making process? If another researcher reviewed your notes and data, could they follow your logic and reproduce your strategy? What information would they need to fully understand your approach?
Yes they would be able to follow along. I have a folder for each weeks work, and python notebook that goes through step by step my working and learnings for each week. 

I do need to spend a bit more time documenting my reasoning and changes each week at the top of the python notebooks! Probably should do that now before it's a pain later.


#### What assumptions are you making in your search/optimisation strategy? Identify at least one key assumption related to the functions or the optimisation process. How might this assumption shape or limit your results?

Our key assumptions are:

- **The functions behave in a “smooth” way** — more like rolling hills than a spiky mess. The model we use to decide where to try next is built with that in mind. If the real function is actually really jagged or jumps around a lot, our guesses for the next point can be off and we might end up refining in the wrong place or getting stuck.

- **The best point is inside the box we’re searching in.** We only look at inputs in a fixed range (e.g. each number between 0 and 1). If the true best point actually sits outside that range, we’ll never find it no matter how many rounds we run.

- **For the higher‑dimension functions (7 and 8), the learned model is a decent stand‑in for the real function.** We use that model to suggest the next point. If the model is wrong in areas we haven’t tried much yet, the suggested point might be a dud.



#### Where do you see gaps or potential biases in your data set? Consider the distribution of your queries, unexplored areas or patterns in how you sampled the search space.




#### What is one significant limitation of your approach? Consider factors such as computational constraints, sampling biases or assumptions about function behaviour that might affect the validity or generalisability of your results.