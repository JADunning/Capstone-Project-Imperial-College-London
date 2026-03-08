#### How does changing the temperature affect token selection and output entropy?
Temperature controls how strongly the model prefers high-probability tokens. Low temperature makes the model choose the safest tokens, lowering entropy. High temperature flattens probabilities, increasing entropy and making less likely tokens more likely to be chosen.


#### Is the output more repetitive or more surprising? What patterns emerge?
Low temperature produces repetitive and predictable output. Higher temperature produces more varied, or "creative" text. The general pattern is a trade-off between stability / expected behaviour at low temperatures and creativity at higher.

#### What changes in output do you observe when adjusting top-p or top-k?
Higher values allow more token options which increases the number of options you get and the variation / creativity. Low values mean you restrict the output to the top expected / most likely choice. 

#### Does limiting the candidate tokens (with top-p) or restricting to a probability mass (with top-k) affect the coherence or creativity in the output?
Yes. Strong limits improve coherence but reduce creativity. Looser limits increase creativity but can reduce coherence. 

#### How do these decoding settings interact with the attention mechanism?
Attention determines which previous tokens influence the next-token probabilities. Decoding settings then control how those probabilities are sampled. So attention shapes what is likely, and decoding settings decide how strictly the model follows those likelihoods.

#### Do certain tokens become more or less likely based on the focus of attention in transformer blocks?
Yes. Tokens related to what the model attends to—such as the subject, recent actions, or grammatical structure—receive higher probability, making them more likely to be selected.

#### What does the visualisation show about how attention guides probability assignment?
It has an equation where Output = attention weight x value. So the model looks at all previous words, assigns each one an attention score, and then multiplies each words value by that score. So words with higher attention weight influence the output more.


#### How do temperature, top-p and top-k sampling help control the model’s behaviour?
They let you balance predictability and creativity. Lower randomness produces safer and more precise output, while higher randomness encourages exploration and variety.

#### When might you want more randomness, and when is precision critical?
More randomness is useful for storytelling, brainstorming, or creative writing. Precision is critical for coding, technical explanations, calculations, or instructions where correctness matters.

#### How does the transformer’s self-attention enable flexible token choice under different decoding schemes compared with older models?
Self-attention is the mechanism that lets each word in a sentence look at all the other words in the same sentence to understand context. This richer context means the model can still choose sensible tokens even when decoding introduces randomness because it can understand the whole phrase, not just words and their close neighbours. 

