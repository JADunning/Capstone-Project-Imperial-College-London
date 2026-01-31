#### Which prompt patterns (zero-shot, few-shot, etc.) did you use, and why? What changed when you simplified vs structured the prompt?

I didn't use any LLMs in my actual BBO code! I did however use Cursor to help with some of the coding.

I mostly used zero shot prompts e.g. "Write a function that loads inputs from the week N text file and returns a numpy array,". Or for the repettivie code week on week I would provide an example of previous weeks code "duplicate this python notebook structure for the current week taking this ____ as an example." which is an example of few-shot prompting. Sometimes I would also role prompt "you are an expert data scientist, review this code and give me suggestions.. "

When you structure the prompt by role prompting or using examples, you definitely do get an output closer to what you want instead of just zero-shot prompting it!


#### What temperature, top-p, top-k and max-tokens settings did you choose? How did they trade off coherence vs diversity? How did they affect your chosen query?

I didn't change the temperature or top-p/top-k directly—the tools I used don't expose those. Although I imagine (and hope) they are using very low temperature for my coding assigninments, I definitely need precision more than creativity.


#### Did token boundaries or unusual input strings affect the model’s behaviour? When did you notice token count limits or truncation influencing the outputs? If no such cases were observed, explain how you checked for those cases.

I don't think unusual input strings affect the models behaviour, I think they've probably tried to account for most of those. You do occasionally hit the token boundary if pasting in large amounts of code, you get a warning and it lets you know, and generally I try to avoid that - I think it's more accurate when you don't overload the prompts.


#### With 17 data points, what limitations did you encounter, such as prompt overfitting, attention focusing on irrelevant context or diminishing returns from longer inputs?
Codex definitely doesn't work well duplicating longer python notebooks! I wonder if it's just not used to working with those as a format, so you do have to watch out when trying to get it to complete longer heavy load tasks.

If you do add a lot of context into your prompts then yes, the LLMs often include irrelevant context or focus on the wrong parts of the context. For example, I asked it to create a new python notebook today with an example, but it then tried editing the current one. I think it just missed the context in amongst the large prompt. 


#### Which strategies did you try to reduce hallucinations? For example, did you use tighter instructions, retrieval of prior relevant information or constrain the output format?

A few strategies:
1. Get the AI to review itself or the code it's written. Especially if you get it to review the code it's written in a new window without the previous context. That I think reduces hallucinations, you're less likely to hallucinate twice.
2. I always try and use few shot prompts, so I will give it example of the code I want it to write.
3. I always provide quite strict instructions for example "build a simple linear regression using this as an example with these parameters"
4. Or I ask it to plan ahead of completing a task using role based prompting "you're an expert CTO designing a new and simple neural network for this problem (add example here).. write out a plan for what you'd need to code step by step to complete this task". Then feed it each step. This reduces the cognitive load on the LLM and the knowledge needed to complete each step in the process. Meaning it often produces far more accurate results. 


#### In future rounds, how would you scale your prompting and decoding strategies when working with larger data sets or more complex LLMs?

I'd:
- Keep my code organised and well structured, avoid large complex functions. Which then helps the LLMs to debug something specific. 
- I'd continue to work in stages. I.e. get the LLM to help me plan first, then do the coding with examples. 
- I keep a folder of pristine code examples, which I can always refer back to in my prompts to improve accuracy.


#### How did these design choices for prompts and decoding help you think like a practitioner balancing exploration, risk and computational constraints in a black-box setting with incomplete information?

I suppose much like prompting where you have to think about what task you need to do, then write a prompt most appropriate for the situation, when working on this black box project you should make sure you understand the data that comes back each week, visualise it, and then come up with a plan that allows you to be most effective / get the best result. 
