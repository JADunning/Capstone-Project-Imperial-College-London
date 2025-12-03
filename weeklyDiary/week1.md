WEEK 1 (W/C 20.11.2025)

### What was the main principle or heuristic you used to decide on each query point (e.g. exploitation of high outputs, exploration of uncertain regions, diversity of samples)?

It’s the first week, and so I wanted to focus on exploration, picking values that were in uncertain regions.

The first thing I did was download and print all the data, then try and visualise it if possible (easy for the 2D and 3D functions).

Then I aimed to pick new points in unexplored spaces.

For function 1 - I just went in the middle, 0.5, 0.5.

For function 2 there seems to be a correlation of high outputs around X1 = 0.65. So I choose to explore that area and went for an X2 value in the unexplored area in the middle of the extreme values on either end (0.5)

For the rest, because they were hard to visualise (and I was running a little bit out of time) I wrote a function that generated 5000 random points, then calculated the distance relative to the other inputs, and then chose the value furthest away from the others. So it picks a value in an unexplored space.

 

### Which function(s) were most challenging to query, and why? What additional information would have helped you?

The most challenging were the high dimension functions, because they’re just hard to get your head around and visualise! It’s hard to form intuitions about where promising regions might lie when I couldn’t visualise the space

What additional information would have helped:

Any simple visualisation aids or basic guidance on which regions of the input space were particularly sparse would be helpful
Any summary statistics about how smooth or noisy each function is would have helped plan an attack.
Slightly more data points would help again, since there’s so many things to consider.



### How do you plan to adjust your strategy in future rounds based on the current performance or uncertainty levels?

I think I need to spend more time getting my head around the multi-dimensional space!

Maybe an algorithmic approach could work well like Bayesien Optimisation to help pick the next steps. So I’d probably work on getting that up and running next week.