---
title: Policy Gradient

---


## Policy Optimization

With policy optimization, we are learning a mapping from a state to the probabilities of actions that should be taken in that state.

This is different from Deep Q-Learning or Tabular Q-Learning. With those methods, we learn what the overall future reward would be if we took a certain action at a certain state and then continued acting optimally from that point onward. After training, the agent can look at those values and decide what action to take depending on how exploitative or explorative it wants to be.

With policy gradients, after training we instead have a policy that directly maps a state to action probabilities. The agent then chooses actions according to those probabilities — either favoring the highest-probability action or sampling from the distribution depending on how exploitative or explorative it wants to be.

The question then becomes: how do we learn this mapping from states to action probabilities?

At the start, all weights and biases are random, so the action probabilities coming out of the policy network are basically gibberish. The policy doesn't know anything yet.

We then start interacting with the environment: observe a state, choose an action, receive an immediate reward, move to the next state, and repeat. This continues for an entire trajectory.

If that trajectory ends up producing a relatively high overall reward, the actions taken throughout it are treated as part of a successful experience, and we want the probabilities of taking those actions in those states to increase.

The difficult part is that we don't know exactly which action contributed most to the success — we only know the trajectory as a whole did well. Because of that, the simple policy gradient idea is to increase the probabilities of all actions that appeared in that successful trajectory.

This is implemented through the loss function. Unlike supervised learning, the loss isn't comparing output to some known correct answer. Instead, it's constructed so that if a trajectory produced a high return, reducing the loss requires increasing the probabilities of the actions taken during it. Once that loss is constructed, backpropagation calculates how weights and biases should change to reduce it. Since the loss was designed correctly, reducing it ends up increasing probabilities for actions associated with successful trajectories and decreasing them for unsuccessful ones.

---

## Actor-Critic

Instead of directly using the Q-value to update the policy, actor-critic uses the **advantage function**, which captures whether taking a certain action at a certain point was better or worse than what the current policy expected at that moment. It's essentially asking: if I took this specific action rather than letting my current policy decide probabilistically, did I do better or worse than expected?

![image](https://hackmd.io/_uploads/r1-0vde-Gl.png)

If we did better, increase the probability of taking that action. If we did worse, decrease it.

![image](https://hackmd.io/_uploads/rk7bFdgbGg.png)

Taking the gradient of that function tells us whether to increase or decrease the probability of certain actions in the future. And once the policy updates, the next state input will produce a new action that may be more accurate for better future outcomes — and we repeat.

What's worth noticing is that with the original policy gradient method, we run a whole trajectory and then multiply by the accumulated reward — essentially a Monte Carlo approach. We wait until we know the true outcome and then update the probabilities of all actions that happened during it. The problem is that every action gets adjusted according to the final outcome regardless of how much it actually contributed.

With actor-critic, we update the probability of the action taken at each state almost immediately after it happens. We take the action, observe the reward and next state, calculate an advantage estimate, and update the policy right away rather than waiting for the entire trajectory to finish. In practice, backpropagation compiles across a batch of data how much each weight and bias should adjust — but that accumulation happens throughout, not just at the end of a single trajectory.

With the TD version of the advantage function especially, most of it almost feels like gibberish other than the immediate reward — because the policy is constantly being adjusted, so the value functions we calculate aren't perfectly accurate; they're just learned estimates. But the immediate reward is accurate because it's what actually happened. Even if the value estimates are noisy, they still help answer the question: "Did this action make things better or worse than expected?" The advantage function gives a rough estimate of how much that particular action contributed, which is much more informative than giving every action credit or blame based solely on the final trajectory reward.

Actor-critic can compute learning signals at each step using bootstrapping (TD error), so it doesn't need to wait for full trajectories before performing updates — though in practice it still updates using batches of collected experience.

---

## PPO

The general idea is to avoid changing the policy too drastically by comparing the ratio of what the new and old policies assigned to a given action, and constraining that change to a certain range.

![image](https://hackmd.io/_uploads/SyRcb3gWfx.png)

PPO can be thought of as trajectory-adjacent because, while it is actor-critic, it clips the magnitude of updates. This makes it work well even when rewards are noisy or inaccurate — such as in VLM-based settings where reward signals come from a model rather than the environment directly.



