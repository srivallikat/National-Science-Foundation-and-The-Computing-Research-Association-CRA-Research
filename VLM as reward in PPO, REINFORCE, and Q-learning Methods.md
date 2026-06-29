---
title: 'VLM as reward in PPO, REINFORCE, and Q-learning Methods'

---

"At the beginning of each episode, the agent is assigned a fixed goal prompt g ∈ G. During interaction, it collects short trajectory segments τt = [It−k+1, . . . , It] of observations, where k denotes the segment length. In the simplest case k = 1, the inferred reward is based only on the most recent observation, while larger values of k allow the reward to depend on a short history of frames."

This confused me at first because, typically with a VLM, we ask a question. However, maybe that question can essentially be considered the goal, since that is what we are trying to answer or evaluate.

Another key takeaway from Masudur's paper was this: why are we consistently trying to define PPO as something trajectory-based? PPO is still an actor-critic method, and actor-critic methods are heavily defined by rewards. So why are we trying to say PPO would not be impacted by a shaky reward signal?

The answer seems to be that even though actor-critic methods are reward-driven, PPO specifically mitigates the effect that a bad reward can have by preventing large changes to the policy. That is why we say PPO is not overly sensitive to shaky, inferred, or imperfectly defined rewards. Even the paper admits this:

"This lemma applies directly to REINFORCE, which performs unbiased Monte Carlo policy gradient updates using complete trajectory returns. While PPO includes a value baseline, the critic is used only for variance reduction. As a result, PPO inherits the robustness properties of trajectory-level methods and remains effective even when inferred rewards are non-Markovian."

The paper recognizes that PPO is not exactly a trajectory-based method, but it clips the effects of what a purely reward-driven update could do. In contrast, when we train DQN, that inferred idea of what the Q-value should output is driven almost entirely by rewards, so those rewards need to be accurate.

"We studied InfeRL, a framework that replaces environment-provided rewards with semantic signals inferred from vision-language models. Our analysis highlighted monotonicity and (quasi-)Markovianity as key properties shaping when such rewards support effective learning."

The VLM does not have to take vision tokens from only the current state. It can take as many previous frames as it wants. This is where the idea comes from that the reward is not solely determined by the current state. That is why we move away from a strictly Markovian process, where the next reward and state are determined only by the current state and action.

"We refer to this formulation as Inference-Based Reinforcement Learning (InfeRL), highlighting its central departure from standard RL: environment-provided rewards are replaced with internally inferred signals."

Again, do not get too confused by this. It is simply saying that the agent is determining the reward signal rather than relying on a reward directly provided by the environment. We are saying the VLM is generating the reward that PPO or REINFORCE will continue to use.

This is also why it makes sense that REINFORCE works well here. We update the probabilities of actions based on the overall trajectory return, so naturally there is less pressure on any individual reward being perfectly accurate.

"To quantify the reliability of inferred rewards, we compute pairwise agreement, Kendall's τ, and Spearman's ρ between rankings induced by inferred returns R̂(τ) and true returns R(τ)."

I like that the paper uses the word reliability here. We are comparing against the true rewards that are given by the environment. The paper argues that we may not want to rely entirely on those rewards because they do not necessarily account for everything we care about, but they still provide a useful benchmark. We can use them to check whether the inferred rewards are at least in the same general ballpark.

Personally, I do not think we strictly need this. If we train a VLM reward model, we are already providing image-text inputs and reward-related outputs during training. However, I understand why they do this comparison. It is mainly an evaluation tool. During actual training, the policy should be learning from the inferred rewards. The true rewards are being used here primarily to evaluate how reasonable those inferred rewards are.

Key takeaway from this paper: use VLM-generated rewards with PPO or REINFORCE. Why? These methods work best when the reward itself is not perfectly accurate but is at least directionally correct.

Do not use DQN. DQN relies much more heavily on reward accuracy because the reward signal is doing most of the work in determining the Q-values that drive learning.

"Policy gradient methods require only trajectory-level monotonicity, making them robust to non-Markovian or noisy inferred rewards. Q-learning methods, however, require both monotonicity and strict (or quasi-) Markovianity of per-step rewards to ensure stability. This explains why trajectory-based methods tend to perform better in practice under inferred rewards, while Q-learning style algorithms are more fragile."

Okay, yes, this is the common-sense part of this entire thing. The only question I have is: what is this paper really trying to say then? We are already aware that for PPO and REINFORCE, the reward from a specific state-action pair does not determine everything. PPO already limits how much a single reward can affect policy updates, and REINFORCE already looks at overall trajectory returns.

This brings me to my main confusion:

"Conceptual framework and theoretical analysis establishes conditions under which inferred rewards support effective learning by emphasizing order-preserving monotonicity of trajectory returns as a key property."

The monotonicity part is what I do not fully understand.

I understand the main ideas. I understand why this becomes quasi-Markovian: the reward is no longer solely determined by the current state and action. I understand why DQN struggles in this setting, since DQN depends much more heavily on accurate rewards. I understand why PPO is more robust because it limits extreme policy updates, and I understand why REINFORCE can tolerate reward noise since it updates action probabilities based on overall trajectory performance.

What I do not understand is why the monotonicity result is being emphasized so heavily.

To be honest, I also did not pay full attention to the experiments, so maybe this idea is more important there and provides a way to evaluate things better.

Do we need to preserve the ordering of rewards so that we adjust the probabilities of actions in the correct direction? And if so, is that not already somewhat common sense? It feels more like an evaluation criterion to make sure we are on the right path rather than a major takeaway on its own.

That is probably the part of the paper I am still missing.
