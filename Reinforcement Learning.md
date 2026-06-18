---
title: Policy Gradient

---

So basically right when we have policy optimization, what we are doing is learning a mapping from a state to the probabilities of actions that I should be taking in that state.

This is different from Deep Q-Learning or Tabular Q-Learning. With those methods, I am learning what the overall future reward would be if I took a certain action at a certain state and then continued acting optimally from that point onward. After training, if I am the agent, I can look at those values and decide what action to take depending on how exploitative or explorative I want to be.

With policy gradients, after training I instead have a policy that directly maps a state to action probabilities. Then, as the agent, I can choose actions according to those probabilities, and again depending on how exploitative or explorative I want to be, I can either favor the highest probability action or sample from the distribution.

The question then becomes: how do we learn this mapping from states to action probabilities?

At the start, all of my weights and biases are random. Since the weights and biases are random, the action probabilities that come out of the policy network are basically gibberish. The policy does not know anything yet.

I then start interacting with the environment. I observe a state, choose an action, and receive an immeadiate reward. Then I move to the next state, choose another action, and receive another immeadiate reward. This continues for an entire trajectory.

Suppose that trajectory ends up producing a relatively high overall reward. In that case, the actions that were taken throughout that trajectory are treated as actions that were part of a successful experience. Therefore, I want the probabilities of taking those actions in those states to increase.

The difficult part is that I do not know exactly which action contributed the most to the success. I only know that the trajectory as a whole ended up doing well. Because of that, the simple policy gradient idea is to increase the probabilities of the actions that appeared in that successful trajectory.

The way this is actually implemented is through the loss function. The loss function is not comparing my output to some known correct answer like in supervised learning. Instead, it is constructed so that if a trajectory produced a high return, then reducing the loss requires increasing the probabilities of the actions that were taken during that trajectory.

In other words, the loss function is designed in such a way that the direction that makes the loss decrease is also the direction that adjusts the action probabilities the way we want. Once that loss has been constructed, backpropagation can calculate how the weights and biases should change in order to reduce the loss. Since the loss was designed correctly, reducing the loss ends up increasing the probabilities of actions associated with successful trajectories and decreasing the probabilities of actions associated with unsuccessful trajectories.

### Actor Critic
But we are saying instead of directly using the Q-value to update the policy, use the advantage function, as it shows whether taking that certain action at that point was better or worse than what our current policy expected at the time. It is basically saying if I did take this action at this moment rather than leaving it up to the probabilities from my current policy, did I do better or worse than expected?![image](https://hackmd.io/_uploads/r1-0vde-Gl.png) What we want to do with this is, if I did better, increase the probability of taking that action. If I did worse, decrease it.
![image](https://hackmd.io/_uploads/rk7bFdgbGg.png)

That function above, if we are trying to increase it's output, taking the gradient actually tells of it tells us to increase or decrease the probability of certain actions happening in the future.

And yes, now that the policy has changed yet again, when we input the next state we will get a new action that might be more accurate to take at the state for better future outcomes, and we can repeat this process again.

What I am noticing is that with the original policy gradient method, we would run a whole trajectory and then multiply by the accumulated reward of that trajectory. In a sense, it was kind of Monte Carlo. We would wait until we knew the true outcome of the trajectory and then update the probabilities of all the actions that happened during it. The problem is that every action gets adjusted according to the final outcome regardless of how much that specific action actually contributed.

But with actor-critic, we are updating the probability of the action taken at that state almost immediately after it happens. We take the action, observe the reward and next state, calculate an advantage estimate, and then update the policy right away rather than waiting for the entire trajectory to finish. Or atleast the backpropagation process in the code can compile for a batch of data how much each weight and bias should adjust in the policy function but even seeing how much each thing should adjust that acculmalation of what should happen is throughout, not just after waiting for that one trajectory to finish, updating the backprop at the end for how much each weight and bias should move, and then that having to happen multiple times bc of a batch before that update happens.

Especially with the TD version of the advantage function, most of it almost feels like gibberish other than the immediate reward. I say gibberish because the policy is constantly being adjusted, so the value functions we calculate are not perfectly accurate and are themselves just learned estimates. However, the immediate reward is accurate because it is what actually happened. Even if the value estimates are noisy, they still help answer the question: "Did this action make things better or worse than expected?" In that sense, the advantage function gives us a rough estimate of how much that particular action contributed, which is much more informative than simply giving every action credit or blame based on the final trajectory reward.

actor-critic can compute learning signals at each step using bootstrapping (TD error), so it doesn’t need to wait for full trajectories before performing updates, but in practice it still updates using batches of collected experience.

## PPO

General idea of this is to try and not change the policy to drastically by comparing ratios of what the new policy and old policy at that action was and letting that chnage only be in a certain range. Unsure of implementation concretely still. 
![image](https://hackmd.io/_uploads/SyRcb3gWfx.png)
Only reason why we can think of PPO as a "trajectory" based method or adjecent is because though it is actor critic, it clips those effects, so this method works well even if rewards are not accurate in VLM settings.



So basically right when we have policy optimization, what we are doing is learning a mapping from a state to the probabilities of actions that I should be taking in that state.

This is different from Deep Q-Learning or Tabular Q-Learning. With those methods, I am learning what the overall future reward would be if I took a certain action at a certain state and then continued acting optimally from that point onward. After training, if I am the agent, I can look at those values and decide what action to take depending on how exploitative or explorative I want to be.

With policy gradients, after training I instead have a policy that directly maps a state to action probabilities. Then, as the agent, I can choose actions according to those probabilities, and again depending on how exploitative or explorative I want to be, I can either favor the highest probability action or sample from the distribution.

The question then becomes: how do we learn this mapping from states to action probabilities?

At the start, all of my weights and biases are random. Since the weights and biases are random, the action probabilities that come out of the policy network are basically gibberish. The policy does not know anything yet.

I then start interacting with the environment. I observe a state, choose an action, and receive an immeadiate reward. Then I move to the next state, choose another action, and receive another immeadiate reward. This continues for an entire trajectory.

Suppose that trajectory ends up producing a relatively high overall reward. In that case, the actions that were taken throughout that trajectory are treated as actions that were part of a successful experience. Therefore, I want the probabilities of taking those actions in those states to increase.

The difficult part is that I do not know exactly which action contributed the most to the success. I only know that the trajectory as a whole ended up doing well. Because of that, the simple policy gradient idea is to increase the probabilities of the actions that appeared in that successful trajectory.

The way this is actually implemented is through the loss function. The loss function is not comparing my output to some known correct answer like in supervised learning. Instead, it is constructed so that if a trajectory produced a high return, then reducing the loss requires increasing the probabilities of the actions that were taken during that trajectory.

In other words, the loss function is designed in such a way that the direction that makes the loss decrease is also the direction that adjusts the action probabilities the way we want. Once that loss has been constructed, backpropagation can calculate how the weights and biases should change in order to reduce the loss. Since the loss was designed correctly, reducing the loss ends up increasing the probabilities of actions associated with successful trajectories and decreasing the probabilities of actions associated with unsuccessful trajectories.

### Actor Critic
But we are saying instead of directly using the Q-value to update the policy, use the advantage function, as it shows whether taking that certain action at that point was better or worse than what our current policy expected at the time. It is basically saying if I did take this action at this moment rather than leaving it up to the probabilities from my current policy, did I do better or worse than expected?![image](https://hackmd.io/_uploads/r1-0vde-Gl.png) What we want to do with this is, if I did better, increase the probability of taking that action. If I did worse, decrease it.
![image](https://hackmd.io/_uploads/rk7bFdgbGg.png)

That function above, if we are trying to increase it's output, taking the gradient actually tells of it tells us to increase or decrease the probability of certain actions happening in the future.

And yes, now that the policy has changed yet again, when we input the next state we will get a new action that might be more accurate to take at the state for better future outcomes, and we can repeat this process again.

What I am noticing is that with the original policy gradient method, we would run a whole trajectory and then multiply by the accumulated reward of that trajectory. In a sense, it was kind of Monte Carlo. We would wait until we knew the true outcome of the trajectory and then update the probabilities of all the actions that happened during it. The problem is that every action gets adjusted according to the final outcome regardless of how much that specific action actually contributed.

But with actor-critic, we are updating the probability of the action taken at that state almost immediately after it happens. We take the action, observe the reward and next state, calculate an advantage estimate, and then update the policy right away rather than waiting for the entire trajectory to finish. Or atleast the backpropagation process in the code can compile for a batch of data how much each weight and bias should adjust in the policy function but even seeing how much each thing should adjust that acculmalation of what should happen is throughout, not just after waiting for that one trajectory to finish, updating the backprop at the end for how much each weight and bias should move, and then that having to happen multiple times bc of a batch before that update happens.

Especially with the TD version of the advantage function, most of it almost feels like gibberish other than the immediate reward. I say gibberish because the policy is constantly being adjusted, so the value functions we calculate are not perfectly accurate and are themselves just learned estimates. However, the immediate reward is accurate because it is what actually happened. Even if the value estimates are noisy, they still help answer the question: "Did this action make things better or worse than expected?" In that sense, the advantage function gives us a rough estimate of how much that particular action contributed, which is much more informative than simply giving every action credit or blame based on the final trajectory reward.

actor-critic can compute learning signals at each step using bootstrapping (TD error), so it doesn’t need to wait for full trajectories before performing updates, but in practice it still updates using batches of collected experience.

## PPO

General idea of this is to try and not change the policy to drastically by comparing ratios of what the new policy and old policy at that action was and letting that chnage only be in a certain range. Unsure of implementation concretely still. 
![image](https://hackmd.io/_uploads/SyRcb3gWfx.png)
Only reason why we can think of PPO as a "trajectory" based method or adjecent is because though it is actor critic, it clips those effects, so this method works well even if rewards are not accurate in VLM settings.



