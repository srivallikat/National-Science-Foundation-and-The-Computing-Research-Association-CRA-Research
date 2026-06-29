---
title: 'Raw Thoughts on SafeVLA: Towards Safety Alignment of Vision Language-Action Model via Constrained Learning'

---

"To tackle this challenge, we make the first systematic explorations into VLA safety alignment. Our approach is grounded in the constrained Markov decision process (CMDP)..."

This immediately confused me. I thought the point of a Markov decision process was that to determine the next action, all you need is the current state. Is that not true? There was another method I looked at where they somehow considered previous states and actions when determining what to do next. Is that what this is referring to?

More than that, in the VLM reward papers, you are already kind of considering additional context when assigning rewards. If the VLM is looking at images or trajectories, then it is not necessarily basing the reward solely on the current state. Alternatively, maybe they are arguing that reward assignment itself is not state-based in the traditional RL sense. Either way, I am confused.

What is even stranger to me is that CMDP seems to be saying that when determining rewards or outcomes, there are secondary considerations involved. In my opinion, that does not necessarily have anything to do with the Markov property itself. You could still have a system where the next state and reward depend only on the current state and action while also introducing additional objectives that you care about.

I honestly think this has more to do with what objective we are trying to optimize than with the Markov assumption itself. The framework can still make decisions using the current state and action, but before selecting an action it considers other functions or constraints. To me, that feels more like something that belongs to the policy or optimization objective rather than to the Markov framework itself.

The paper then says:

"...leveraging methodology from safe reinforcement learning (SafeRL) for optimization."

I have to admit that I do not really know what SafeRL is yet.

The paper continues:

"We investigate an integrated safety approach (ISA), which systematically considers four key aspects: comprehensively modeling safety requirements within the CMDP setup, actively eliciting diverse unsafe behaviors to inform constraints, rigorously constraining VLA policies using CMDP-compliant SafeRL techniques, and thoroughly assuring safety through targeted evaluations."

The first part is interesting. Why would a Markov process have anything to do with modeling safety requirements? After reading a bit more about CMDPs, it starts to make some sense. The entire point seems to be introducing additional functions or constraints into the decision-making process. That interpretation makes sense to me, although it still does not match what I traditionally think of when I hear "Markov."

The part about constraining VLA policies also confused me.

When I hear "policy" in RL, I think of a probability distribution over actions. Here, when they talk about constraining a VLA policy, it feels like they mean something broader than that. Maybe they still technically mean the same thing, but in practice it feels different because the policy is now an entire vision-language-action model. The action expert is supposed to output what actions to take for the next 50 steps, not a probability.

The paper then says:

"The core insight of such an approach is to explicitly trade off safety and task performance, prioritizing safety adherence."

This reminds me a lot of Gloria's project. There was also a balance between maximizing performance and maintaining safety constraints.

The paper later introduces Safety-CHORES:

"Thus, we propose Safety-CHORES to comprehensively assess safety alongside task performance in complex, procedurally generated environments."

Does our VLM not only return rewards but ouputs for this constrants we need to consider is my big question that I'm also confused about?

One thing that stood out to me is that prior work such as FLaRe and GRAPE also used RL fine-tuning on VLAs, but their goal was task performance and generalization. This paper is arguing that safety constraint satisfaction should be treated as a first-class objective instead.

One realization I had while reading this is that the action expert is operating at a specific state and predicting future actions. Those predicted actions are being generated for some objective that was learned previously.

This entire paper basically feels like a safety framework that could potentially be applied to our setup as well.

The main thing I think I understood is that CMDP allows additional functions or constraints to be considered alongside reward maximization.

So what do I think I understand so far?

The architecture we are discussing seems to work like this: the VLA is initially trained in a supervised fashion where we know the desired outputs. Then, after the VLA takes an action, we ask a VLM whether that action was good or bad. The VLM assigns a reward. That reward is then somehow used to update the action expert.

This is where I become completely lost.

How exactly is the action expert being updated?

The action expert is essentially predicting a chunk of future actions. In some implementations it is predicting dozens of actions ahead. Reinforcement learning updates probabilities, so where exactly is that update happening inside the action expert? The paper does not seem to explain this at all.

Then this paper starts talking about CMDPs and safety constraints, and I feel like I am missing a major piece of the puzzle.

Right now, my interpretation is that the paper is saying reward assignment should incorporate CMDP-style constraints. Fine.

But where exactly is the action expert in this loop?

How is it being modified?

That is still unclear to me. Is it directly just going to make an action pop up more and change weights inside the action expert somehow I'm not sure.

Reinforcement learning updates policies, which I understand as action probabilities. A VLM assigning rewards just changes how rewards are generated. Whether we use PPO, DQN, or something else, the reward assignment mechanism changes, but the RL algorithm still has to update the policy somehow, meaning it has to update the action expert. This paper does not go into that.

That is the part I do not understand.

I am also confused about the constraints.

Do we need the environment to return separate signals for the constraints?

My current understanding is that a cost function is simply another function we define. Depending on what we care about, we can maximize reward while minimizing cost. If that is true, then it seems like the constraints do not necessarily have to come directly from the environment. They could come from some separate safety evaluator as well.

Key takeaway from this paper: use CDMP.

