---
title: Deep Q Learning

---

**Step 1: Feed in the current state**

Suppose we are in state s.

The network outputs (the model with gibberish weights and biases that predicts the Q-values for all possible actions given a state):

network(s) = [2, 5, 1]

which means:

Q(s, Left) = 2

Q(s, Right) = 5

Q(s, Jump) = 1

The agent chooses:

Right

because 5 is the largest. This is also the agent for the current action being greedy. But remember, later on, we also have to be able to get accurate values for Left and Jump so our policy here does not honestly matter.

In our equation that helps us update our current state-action pair, we need to see the value of the next state-action pair to determine if the action we took was actually any good. The following steps below are how we do that.

**Step 2: Environment responds**

We execute Right and receive:

reward = 3

next_state = s'

This goes back to the idea that the environment, whether it is trained through supervised learning to give the reward and next state or whether it is just hard-coded (in many scenarios it is just hard-coded), is what gives back the next state and reward.

A lot of the time, the reward is hard-coded based on what the environment says the next state is.

For example:

```python
def step(action):

    next_state = transition(state, action)

    if next_state == HOLE:
        reward = -100
    elif next_state == GOAL:
        reward = 100
    else:
        reward = -1

    return next_state, reward
```

So the environment is responsible for producing the next state and reward.

Now we can feed the next state into our network because we found out what the next state would be from the environment.

Suppose:

network(s') = [10, 4, 7] (the values inside are the possible Q values)

Since we are using a greedy policy for training, we assume that in the future we will take the action with the highest Q-value estimate. In this case, that value is 10. Since we are greedy here again, this would be considered on-policy. We we were not greedy here as well at the same level, this would be considered off-policy.

So our Bellman target becomes:

y = reward + γ * max(Q(s'))

If γ = 0.9:

y = 3 + 0.9(10)

y = 12

Notice the special part here. Even though the network predictions are basically gibberish at the beginning, the reward is real. The reward is the accurate part that anchors everything and is ultimately what makes all of this work.

Now we have found our target. This is what we want our weights and biases to slowly retrofit toward.

**Step 3: Create the training target**

Now this is the important part.

For our original state, the network predicted:

[2, 5, 1]

But what prediction are we actually trying to improve?

We chose the action Right because it had the highest value of 5. Therefore, the only output we are directly updating is the one corresponding to Right.

We replace:

[2, 5, 1]

with:

[2, 12, 1]

So what we are training the network to do is:

Input:
s

Desired Output:
[2, 12, 1]

Basically, over time, we will slowly improve each state action pair that the state gives and for all the states, but it is done like this one at a time.

**The important thing to notice is that the data we are retrofitting the network to is not just the scalar value 12. The training target for this example is actually:

[2, 12, 1]**

**Step 4: Update the network**

Now the network adjusts its weights and biases through backpropagation so that when we input state s, the output becomes a little closer to:

[2, 12, 1]

Maybe after one update it becomes:

[2.1, 6.2, 1.0]

Then later:

[2.0, 9.7, 1.1]

And eventually something close to:

[2.0, 12.0, 1.0]

The exact values of the weights and biases and how manyy layers do not matter. It is funny even becuase before we can think of the Q value as a sum of the rewards for all optimal actions at each state but now its just what weights and biases have retrofitted to. Yes the meaning is still that but thats not exactly how we got to the value.

**Other Thoughts on This**

It is very cool that we do not need accurately labeled data telling us the true Q-value for every state-action pair. That would basically be impossible to obtain.

Instead, the beautiful part is that we know the reward is real.

Just like in tabular Q-learning, we may start with completely wrong estimates. In tabular Q-learning we often initialize everything to 0. In DQN, we start with random weights and biases. In both cases, the reward is what grounds the learning process and slowly pushes the estimates toward something meaningful.

Another thing I finally understand is that the code for determining the next state and reward is usually built into the environment. The reinforcement learning algorithm itself does not know how the world works. It simply asks the environment:

"If I take this action at the current state I'm in, what happens next?"

and the environment responds with:

(next_state, reward).

And often in tabular Q-learning, after we update one cell in the Q-table, we need to know what state we ended up in so that we know what state-action pair we are dealing with next. That information comes from the environment. Whether we are using tabular Q-learning or DQN, the environment is responsible for giving us the next state and reward.

Another thought I have is about on-policy and off-policy learning.

In our DQN example, when computing the target, we assumed that in the next state we would choose the action with the highest predicted Q-value:

y = r + γ max_a' Q(s', a')

That is interesting because our model starts off as complete gibberish. We are trusting our own estimates to decide what the "best" future action is, even though those estimates may not be very good yet.

This made me wonder whether sometimes it would make sense to use a less optimal action in the next state when computing the target. Maybe because our estimates are still poor, being less strict about always taking the maximum could help us discover better strategies.

This is related to the distinction between off-policy and on-policy learning.

Our DQN example was off-policy because when computing the target we assumed that in the future we would always take the action with the highest predicted Q-value, regardless of what action we actually end up taking.

By contrast, SARSA is an on-policy algorithm. Instead of using the maximum predicted Q-value in the next state, it uses the action that was actually selected by the current policy. If the policy includes exploration, then the update includes that exploration as well.

One subtle thing I learned is that exploration and on/off-policy learning are not exactly the same thing. DQN can still explore using something like ε-greedy action selection. The reason it is considered off-policy is not because it explores, but because its update rule assumes that future actions will be chosen greedily. SARSA is considered on-policy because its update rule uses the action that was actually chosen by the policy.

When implementing a model like this, two of the main things I would experiment with are:

1. The neural network architecture itself. Since the Q-function is being represented by a neural network, I would try different layer sizes, depths, and architectures that work well for the type of state representation I have.

2. The exploration versus exploitation tradeoff. If I explore too little, I might miss important strategies. If I explore too much, I might spend too much time taking bad actions. Finding the right balance seems like one of the most important hyperparameters in reinforcement learning. naw this isn't possible. DQN works on choosing the max next one I have heard. But in code I also see how what I was saying is possible.

What I find fascinating is that even though the network starts out with random weights and produces essentially meaningless predictions, the reward signal and repeated interaction with the environment gradually shape those weights into a useful approximation of the Q-function.
