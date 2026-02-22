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
1. The functions behave in a “smooth” way. e.g. more like rolling hills than a spiky mess. The model we use to decide where to try next is built with that in mind. If the real function is actually really jagged or jumps around a lot, our guesses for the next point can be off and we might end up refining in the wrong place or getting stuck.
2. The best point is inside the box we’re searching in. We only look at inputs in a fixed range (e.g. each number between 0 and 1). If the true best point actually sits outside that range, we’ll never find it no matter how many rounds we run.
3. For the higher‑dimension functions (7 and 8), the learned model is a decent stand‑in for the real function. We use that model to suggest the next point. If the model is wrong in areas we haven’t tried much yet, the suggested point might be wrong.


#### Where do you see gaps or potential biases in your data set? Consider the distribution of your queries, unexplored areas or patterns in how you sampled the search space.
In terms of my own exploration, I've probably ended up with most of my later queries bunched near where the model thought the best point was — so we've got loads of points in those "promising" regions and far fewer in the rest of the space. That means whole chunks of the search space (e.g. corners or edges) might be barely touched, and if the true optimum was hiding in one of those under-sampled areas we'd have missed it. The bias is basically: we trust the model to point us to good spots, so we keep sampling there and don't spread out as much as we might if we were doing purely random search.



#### What is one significant limitation of your approach? Consider factors such as computational constraints, sampling biases or assumptions about function behaviour that might affect the validity or generalisability of your results.

The most significant limitation in our approach is the fact we have to only do 1 submission per week. That means we're locked into a single choice — we can't try a few different strategies and see which works best, and if we're wrong we have to wait a whole week. In practice that can make us more conservative and favour "safe" picks or functions we already think are doing well, so the results might be biased toward what we've already committed to rather than a more even exploration.

Another limitation is time: we're doing this alongside other commitments, so I can't fully optimise every single function each week. I end up picking 2 to 3 functions to focus on that show the most potential for improvement. The downside is that the other functions get less attention, so improvement is uneven across the set and we might be missing gains on the ones we deprioritise.