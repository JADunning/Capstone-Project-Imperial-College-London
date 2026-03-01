#### How have patterns in your past queries influenced your latest choices?
I have categorised every output into improving or not improving.. those algorithms where I am not improving I spend a bit more time on, change something about my approach, and try to ensure I do make week on week improvements. 

I also have changed the approach now based on the type of function, for example functions with more noise, I use a more noise-appropriate BO technique. 


#### Have you identified any ‘clusters’ or recurring regions in your search space that seem promising? (Explain your reasoning – whether you have identified such regions or not.)
Yes, for the most part, my technique has been explorative BO for the first few weeks before then switching to exploitative BO. This has targeted areas around the highest value, which can be classified as a promising search space. 


#### Which strategies or parameter choices have proven less effective, and how are you adjusting for them?

Several choices in my previous exploitative BO alhorithm proved less effective on Function 2, so I switched to a new BO alghorithm this week and adjusted it as follows:

What didn't work:
1. Fitting the GP on every observation with no duplicate handling 
   When the same or very similar inputs were evaluated in different weeks, the GP saw multiple x-points with different y-values. Treating them as distinct forced the mean to fit through all of them, which created unrealistic spikes in the GP surface and unstable next-point suggestions.

2. No explicit observation noise
   The GP used only RBF + constant kernel and a small fixed `alpha`. Repeated evaluations at nearly the same x were effectively treated as noiseless, so the model was overconfident and the predicted mean could blow up in artefact regions. That made the “expected value” for the proposed row implausible (and led to relying on output clipping as a workaround).

3. Global maximization of the GP mean 
   The acquisition step did a global candidate scan over the whole search space and then local refinement. The optimiser could therefore chase global spikes in the mean that were due to noise or duplicate artefacts, suggesting points far from the current best (incumbent) and wasting evaluations.

4. Output clipping
   Clipping the proposed row’s output value to a “plausible” range was a band-aid for the above; it hid implausible GP predictions instead of fixing the model.

What I adjusted this week:

- Duplicate aggregation: Before fitting the GP, I merge duplicate and near-duplicate inputs into clusters, use the cluster centre as x and the mean of the y-values as the target. That reduces spurious peaks from repeated evaluations at similar x.

- Noise-aware GP: I added a WhiteKernel so the GP models observation noise explicitly. Repeated evaluations no longer create unrealistic spikes; the model learns a noise level and smooths appropriately.

- Local trust-region exploitation: I no longer maximize the GP mean globally. The next point is chosen by maximizing the mean only in a trust region around the incumbent. That keeps suggestions near the current best and avoids chasing unstable global spikes.

- Raw GP mean for the proposed row: I removed output clipping; the “Output value” for the proposed row is the raw GP predicted mean at the suggested point. With aggregation and a WhiteKernel, this stays in a plausible range; the notebook still warns if it falls outside the observed range so I can sanity-check.

In short: the previous BO was too sensitive to repeated evaluations and global artefacts. The new algorithm is adjusted for noisy, weekly data by aggregating duplicates, modelling noise, and restricting the search to a local trust region around the incumbent.


#### In what ways do your refinements parallel how clustering algorithms separate meaningful patterns from noise?

I treat the data as structure plus noise in a similar way. I cluster near-duplicate observations and use the cluster centre and mean output before fitting the GP, so the pattern is the representative point and the noise is the spread that gets averaged and then modelled by the WhiteKernel. I also only search in a trusted region near to the current best point, and that is like focusing on a dense, promising region and ignoring distant spurious peaks as noise instead of chasing them.

#### If your query results were plotted, what trends or groupings might appear? How could these inform your next iteration?

If I plotted output value (or best-so-far) against week for each function, I’d expect to see upward trends where the approach is working and flatter or noisier series where it isn’t. Groupings might appear by function: some functions improving steadily, others plateauing or stuck, and possibly a cluster of noisier functions where results jump around more. That would tell me where to keep the current strategy and where to try a different one in the next iteration.