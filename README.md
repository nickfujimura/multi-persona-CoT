# multi-persona-CoT
A proof of concept for a multi persona LLM discussion framework which dynamically spins up different perspectives to tackle a problem presented

I’ve been thinking about scaling laws and I think scaling different persona’s for thinking about hard problems could contribute to “smarter” agents. It’s been said that you may have 1000 Einstein’s in your coffee maker to make you the perfect cup of coffee in the morning, however, I’ve found that even with stochastic responses, agents in a specific perspective can get stuck on certain bugs or loop problems. In human organizations, we view diverse perspectives as an asset that can bring multiple viewpoints to a problem, even when those viewpoints are in tension. E.g. if you want to grow but manage risk at the same time. Humans are bad at holding conflicting priorities in their minds while addressing problems and largely we have success in having people take on specific roles and focus and then hash out the problem. With LLMs, we have the opportunity to spin up nearly limitless perspectives and have them all contribute to the conversation! 

I’ve vibe coded a small example project that utilizes this idea to take a prompt, analyze and dynamically create the different (ideally divergent!) perspectives that might be useful in answering the prompt, and then have them participate in a “discussion” where each proposes a solution, they critique each other's' proposals, and they iterate until they achieve consensus. So far I’ve gotten good results! 

One of my go-to evals is to ask a model how tire pressure relates to the distance traveled in one full rotation of a wheel in idealized circumstances. So I gave the model the following prompt: 

>What is the impact of a change in tire pressure on the distance traveled per rotation of the tire? Assume no slippage of the tire on the surface or the rim and assume the tread is a fixed length which does not expand or contract with pressure changes because of the steel belt in the tire tread. Use various engineering and scientist personas as well as a contrarian first principles thinker to answer

Most models (with the notable exception of GPT-5 will tell you that lower pressure results in a smaller radius, and therefore a smaller circumference. So lower pressure → lower distance traveled per rotation. GPT-5 will tell you that if the outer tread is a fixed length, the wheel travels the same distance regardless because it travels the exact length of the tread. 

I’ve been using GPT-OSS 20B locally to run my experiments, and I’ve found that it will usually come to the “lower pressure → lower distance” conclusion but that if 1 persona comes to the “it’s the same distance” conclusion at the outset, the group will (sometimes begrudgingly) come to the “it’s the same distance” conclusion collectively. 

I’d be curious if this “multi persona chain of thought” approach results in different outcomes vs. what you would get with a single persona thinking model. Feel free to download the code and try it with your favorite model, though I’ve only tested Gemini and GPT-OSS (via ollama) so far so YMMV. 
