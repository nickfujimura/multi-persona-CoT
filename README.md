# multi-persona-CoT
A proof of concept for a multi persona LLM discussion framework which dynamically spins up different perspectives to tackle a problem presented

I’ve been thinking about scaling laws and I think scaling different persona’s for thinking about hard problems could contribute to “smarter” agents. It’s been said that you may have 1000 Einstein’s in your coffee maker to make you the perfect cup of coffee in the morning, however, I’ve found that even with stochastic responses, agents in a specific perspective can get stuck on certain bugs or loop problems. In human organizations, we view diverse perspectives as an asset that can bring multiple viewpoints to a problem, even when those viewpoints are in tension. E.g. if you want to grow but manage risk at the same time. Humans are bad at holding conflicting priorities in their minds while addressing problems and largely we have success in having people take on specific roles and focus and then hash out the problem. With LLMs, we have the opportunity to spin up nearly limitless perspectives and have them all contribute to the conversation! 

I’ve vibe coded a small example project that utilizes this idea to take a prompt, analyze and dynamically create the different (ideally divergent!) perspectives that might be useful in answering the prompt, and then have them participate in a “discussion” where each proposes a solution, they critique each other's' proposals, and they iterate until they achieve consensus. So far I’ve gotten good results! 

One of my go-to evals is to ask a model how tire pressure relates to the distance traveled in one full rotation of a wheel in idealized circumstances. So I gave the model the following prompt: 

>What is the impact of a change in tire pressure on the distance traveled per rotation of the tire? Assume no slippage of the tire on the surface or the rim and assume the tread is a fixed length which does not expand or contract with pressure changes because of the steel belt in the tire tread. Use various engineering and scientist personas as well as a contrarian first principles thinker to answer

Most models (with the notable exception of GPT-5 will tell you that lower pressure results in a smaller radius, and therefore a smaller circumference. So lower pressure → lower distance traveled per rotation. GPT-5 will tell you that if the outer tread is a fixed length, the wheel travels the same distance regardless because it travels the exact length of the tread. 

I’ve been using GPT-OSS 20B locally to run my experiments, and I’ve found that it will usually come to the “lower pressure → lower distance” conclusion but that if 1 persona comes to the “it’s the same distance” conclusion at the outset, the group will (sometimes begrudgingly) come to the “it’s the same distance” conclusion collectively. 

I’d be curious if this “multi persona chain of thought” approach results in different outcomes vs. what you would get with a single persona thinking model. Feel free to download the code and try it with your favorite model, though I’ve only tested Gemini and GPT-OSS (via ollama) so far so YMMV. 

I've included a log of the Tire Pressure eval run where a single "outlier agent" is able to convince 4 other agents of the view that because the tread length is constant, the distance traveled is independent of tire pressure. A summary of that discussion is provided below: 

Based on the provided debate log, here is a summary of the discussion regarding the effect of tire pressure on travel distance per rotation under idealized conditions.
## Summary of the tire discussion
### The Core Question

The debate centered on a theoretical problem: "What is the impact of a change in tire pressure on the distance traveled per rotation of the tire?". This was to be answered in an idealized environment with two key assumptions:
1.  There is **no slippage** of the tire on the road or the rim.
2.  The tire tread has a **fixed length** that does not stretch or contract due to a steel belt.

---

### Initial Positions

The five agents began the debate with conflicting conclusions, each approaching the problem from their specialized viewpoint.

* **Mechanical Engineer, Materials Scientist, Thermodynamic Analyst, and Mathematician:** These four agents initially concluded that **increasing tire pressure increases the distance traveled per rotation**. Their shared reasoning was that higher pressure makes the tire stiffer, reducing the amount it flattens under a load. This decrease in "flattening" or deflection results in a larger effective rolling radius, which in turn increases the circumference and the distance covered per revolution.

* **Physicist – Rotational Kinematics:** This agent was the initial outlier, arguing that a change in tire pressure has **no impact** on the distance traveled per rotation. The core of this argument was that if the steel belt fixes the tread's length, then the tire's outer circumference is also fixed, making the distance per rotation constant regardless of pressure changes.

---

### The Path to Consensus

The debate evolved over several rounds as the agents analyzed each other's models. Initially, the idea that pressure affects the tire's "flattening" and thus its effective radius dominated the discussion.

However, the Physicist's argument gradually gained traction. The crucial realization was that while pressure does change the *shape* of the tire and the *geometry* of the contact patch, it cannot change the fundamental length of the inextensible steel-belted tread.

By the third round, several agents, including the Materials Scientist and the Physicist, converged on the idea that since the tread length ($L_t$) is constant, the effective radius ($R_{\text{eff}} = L_t / 2\pi$) must also be constant. Any deformation of the sidewalls simply alters how that fixed length meets the road, but not the length itself.

---

### Final Consensus

The agents reached a unanimous consensus after four rounds.

The final conclusion is that, within the strict idealized constraints of the problem, a **change in tire pressure has no effect on the distance traveled per rotation**.

The definitive reasoning is that the distance covered in one perfect, no-slip rotation is precisely the circumference of the tire's tread. Because the problem assumes an inextensible steel belt, this tread length is a fixed constant. Therefore, regardless of how pressure deforms the tire's sidewalls or alters the shape of the contact patch, the fundamental distance rolled out in one turn cannot change.
