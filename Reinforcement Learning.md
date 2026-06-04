# Unifying My Reinforcement Learning Understanding


---

# PART 1 — The actual environment

Reality itself is just:

![image](https://hackmd.io/_uploads/B17TtsPlzg.png)

Meaning:

Current state:

* hungry
* stressed
* low sleep

Action:

* fried chicken

Then reality produces:

* immediate reward
* next state

Example:


![image](https://hackmd.io/_uploads/SkekcoPxMg.png)



---

# PART 2 — The agent does NOT magically know that the reward would be +8 or that the next state would be sluggish at the gym



---

# Learned reward model

The agent may learn:


![image](https://hackmd.io/_uploads/rJ4ncsveGe.png)

which predicts immediate reward.

Example:

| State  | Action        | Predicted Reward |
| ------ | ------------- | ---------------- |
| hungry | fried chicken | +8               |
| hungry | salad         | +5               |

This can be trained from experience data like:

| state  | action        | observed reward |
| ------ | ------------- | --------------- |
| hungry | fried chicken | 8               |
| hungry | salad         | 5               |

So **supervised-style** learning can absolutely exist inside RL systems.

---

# Learned transition model

The agent may ALSO learn:

![image](https://hackmd.io/_uploads/B1Z-ijvlGe.png)


Example:

| State  | Action        | Predicted Next State |
| ------ | ------------- | -------------------- |
| hungry | fried chicken | sluggish gym         |
| hungry | salad         | energized gym        |

Again this can be learned from experience trajectories.

This led me to another realization:

if I observe “sluggish gym,” that state actually DOES implicitly contain information about previous actions because the model learned that fried chicken often leads there.

So yes:

* sluggish gym statistically encodes evidence about the previous food choice.

However, the Markov idea is more subtle than:

> “the state contains no history.”

Instead, the idea is:

> the current state contains enough summarized information from the past so that I no longer need the explicit past history to predict the future.

So once I arrive at:


![image](https://hackmd.io/_uploads/Bkmzvnwefl.png)



for the NEXT decision and NEXT prediction, I may only need:

* current energy
* fatigue
* digestion state
* etc.
(this is all included in sluggish gym)
So, Even though those variables were CAUSED by the earlier action, which shows that the earlier action still matters,
but it matters in how it changed the current state. To go to the next state, what I need is only my current state.

That is the Markov property: to go to the next state you only need the previous one, not the previous one plus the previous previous one explicitly.

---

# PART 3 — Where value functions enter

The value function is the agent’s prediction of long-term future goodness at the moment.

Suppose the agent currently believes:

| State         | Estimated Future Value |
| ------------- | ---------------------- |
| energized gym | 7                      |
| sluggish gym  | 2                      |

These are not intrinsic truths.

They mean something like:

![image](https://hackmd.io/_uploads/SJa52jwefx.png)


Meaning:

> “If I end up energized and continue following policy (\pi), I expect good future outcomes.”

And:

![image](https://hackmd.io/_uploads/r1Sh2jDeMe.png)


means:

> “If I end up in the sluggish state and sometimes choose the most optimal actions and sometimes choose exploratory actions (following the policy)as I go through the states, future outcomes are worse.”

#### What is Policy? 
Is is knowing what action to take at a state. A policy based method comes in handy for determining value function.

So basically right when we have policy optimization, what we are doing is learning a mapping from a state to the probabilities of actions that I should be taking in that state.

This is different from **Deep Q-Learning or Tabular Q-Learning**. With those methods, I am learning what the overall future reward would be if I took a certain action at a certain state and then continued acting optimally from that point onward. After training, if I am the agent, I can look at those values and decide what action to take depending on how exploitative or explorative I want to be.

With policy gradients, after training I instead have a policy that directly maps a state to action probabilities. Then, as the agent, I can choose actions according to those probabilities, and again depending on how exploitative or explorative I want to be, I can either favor the highest probability action or sample from the distribution.

#### The question then becomes: how do we learn this mapping from states to action probabilities?

At the start, all of my weights and biases are random. Since the weights and biases are random, the action probabilities that come out of the policy network are basically gibberish. The policy does not know anything yet.

I then start interacting with the environment. I observe a state, choose an action, and receive an immeadiate reward. Then I move to the next state, choose another action, and receive another immeadiate reward. This continues for an entire trajectory.

IMPORTANT Suppose that trajectory ends up producing a relatively high overall reward. In that case, the actions that were taken throughout that trajectory are treated as actions that were part of a successful experience. Therefore, I want the probabilities of taking those actions in those states to increase.

The difficult part is that I do not know exactly which action contributed the most to the success. I only know that the trajectory as a whole ended up doing well. Because of that, the simple policy gradient idea is to increase the probabilities of the actions that appeared in that successful trajectory.

The way this is actually implemented is through the loss function. The loss function is not comparing my output to some known correct answer like in supervised learning. Instead, it is constructed so that if a trajectory produced a high return, then reducing the loss requires increasing the probabilities of the actions that were taken during that trajectory.

IMPORTANT In other words, the loss function is designed in such a way that the direction that makes the loss decrease is also the direction that adjusts the action probabilities the way we want. Once that loss has been constructed, backpropagation can calculate how the weights and biases should change in order to reduce the loss in batches. Since the loss was designed correctly, reducing the loss ends up increasing the probabilities of actions associated with successful trajectories and decreasing the probabilities of actions associated with unsuccessful trajectories.


**What do I love about the value function?** It includes the weightage of the chance for both options. At that state, continuing from there, yes the idea is even at that state we don't know what we are choosing, but it looks like this: ![image](https://hackmd.io/_uploads/H13kU_sxfl.png)
![image](https://hackmd.io/_uploads/ry5w8usgzg.png)

You can see in the Bellman Expectation Equation that we are doing that fraction times both action options. If we did the same thing but for the optimal policy, we would choose between 5 + the value of energetic or 8 + the value of sluggish, which would allow the equation to account for Q(action, hungry) because we are accounting for exactly which action we chose by not adding both of them in there. ![image](https://hackmd.io/_uploads/r1RPi3nxGe.png)
![image](https://hackmd.io/_uploads/r1B-62hxfg.png)
Really, so much of the magic happens because you know the immediate reward of the state. So even though we initialize everything with 0, as we go forward and train, it is because of the reward that the updates can happen.

Also, the question we need to ask is: for the next state that we go to, when we choose the max from there, at the start we have all 0s, right? So then what happens when we have to choose from that? Is that what the other training policy is about?

For my current Q(state, action), do I choose it based on which action has the greatest value at the state I'm currently at in the table? And then for training, does the next state and action that I update with have to be chosen based on what is optimal or not?

Well, not always for the current state. Here you kind of have to go through everything. But for the future state, if you choose to be optimal and pick the greatest value in the table there as well, that's off-policy because we don't actually know what the agent would do there in practice.

This concept of TD seems to be mainly used with Q-learning. 

The discount factor doesn’t decide what actions are good or bad—that’s the policy’s job.

The discount factor is not what decides that behavior. It only controls how much future rewards matter when computing the current value.

That said, it still has an indirect effect, because the value function is updated using future rewards. So when we back up information from future states, the discount factor controls how much those future outcomes influence the current estimate.


---

# PART 4 — Combining models + values

Now the agent can mentally simulate outcomes.

Current state:

![image](https://hackmd.io/_uploads/H1p-psDgGe.png)

---

## Option A — Fried chicken

Reward model predicts:

![image](https://hackmd.io/_uploads/Hy47piPefl.png)


Transition model predicts:

![image](https://hackmd.io/_uploads/SkbVTsPefe.png)


Value function predicts:

![image](https://hackmd.io/_uploads/B1xB6svgGx.png)


Estimated long-term return:

![image](https://hackmd.io/_uploads/ByxLpivgzx.png)


---

## Option B — Salad

Reward model predicts:

![image](https://hackmd.io/_uploads/rJF6TsPeMe.png)


Transition model predicts:

![image](https://hackmd.io/_uploads/rkmA6iDgfg.png)


Value function predicts:

![image](https://hackmd.io/_uploads/SkA06ovxfe.png)


Estimated return:

![image](https://hackmd.io/_uploads/S1mlCivxMe.png)


This may end up larger overall even though the immediate pleasure reward was smaller.

---

# Exploration vs exploitation

Another realization I had:

even if the current estimates suggest salad has larger long-term value,
the agent may STILL choose fried chicken sometimes.

That is exploration.

The agent is not always greedily choosing the current best estimate.

Sometimes it intentionally tries:

* uncertain actions
* less-valued actions
* less-explored trajectories

because its current estimates may still be wrong.

So even if:

![image](https://hackmd.io/_uploads/HyvwCovgMl.png)


the agent might still choose fried chicken occasionally to gather more information.

This is the exploration-exploitation tradeoff.

---

# PART 5 — Where TD learning fits

Now suppose I ACTUALLY choose salad.

Transition:

![image](https://hackmd.io/_uploads/H1_KCivlGg.png)


Then TD learning says:

> “If energized gym predicts good future outcomes, then hungry should inherit some of that goodness because it led there.”

Actually, for this, I kept having a misunderstanding that we needed to start at the last state and work backward. But no—if we look at the example of tabular Q-learning, we can see that if we initialize everything to 0, it is the reward that drives the updates. Because of that, it doesn't matter what direction we go in. As the agent experiences rewards and updates its estimates, the information gradually propagates through the table.

So we update:

![image](https://hackmd.io/_uploads/B1-jAiDeMl.png)


The future estimate:

![image](https://hackmd.io/_uploads/HJOhRjPxzx.png)

is used to update the earlier one.

This is bootstrapping:

* using current future estimates
* to improve earlier estimates.



---Monte Carlo![image](https://hackmd.io/_uploads/rk5Dk9sxMx.png)
The photo is monte-carlo learning style. Instead of using the value that comes after that state to nudge the value at that state accordingly, It goes through multiple states and steps for real, it calculates the total reward, and updates what the value should be accordingly. 

Again remember, for all of this, the pairing of rewards to actions + at what state is already learned!

# The overall pipeline

The full RL reasoning pipeline becomes:

Current state:
[
s
]

↓

Choose action:
[
a
]

↓

Reward model predicts:
![image](https://hackmd.io/_uploads/rJO1y3vlfl.png)
(this can be hard coded or can be supervised learning, but either way the envirnoment gives the reward)

↓

Transition model predicts:
![image](https://hackmd.io/_uploads/H1ag1hDxzg.png)
(we are saying supervised learning here).The states in a set are all possible states, it is not written out that here are the states in order, that defeats the point of saying you need both state and action to move on to the next state.


↓

Value function predicts:
![image](https://hackmd.io/_uploads/Bk2bJnDxzx.png)


↓

Estimate long-term return:
![image](https://hackmd.io/_uploads/rJ3z12Dxze.png)


↓

Use this to:

* evaluate value at that state (this means we don't know the action at the state itself we are choosing)
* choose action based on greed or explore (choose based on the policy set)
* to to next state and see the value there (this is the "evalute what action" step again)
* update previous value accordingly
*But what is the point of this kind of value function if we are just going to act according to policy each time and the reward comes from the environment? It's like the training for the policy is based off just taking actions, and if that led to something with high reward, then adjusting the probabilities corresponding to those actions and for those activation numbers, then nudging the weights and biases appropriately for that. I'm not sure.


---


