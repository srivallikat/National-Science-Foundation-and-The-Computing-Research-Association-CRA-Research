---
title: APF Fields - DDPG (Deep Deterministic Policy Gradient) - Support for VLM in preference ranking

---


**Artificial Potential Field**
*In subject area: Computer Science*

An artificial potential field is a path-planning algorithm that uses attractive and repulsive forces to guide a UAV through an environment, with attractive forces pulling the UAV towards certain points and repulsive forces pushing it away from obstacles, enabling collision-free navigation.

**The Problems with Traditional APF**
While the original APF is fast and computationally lightweight, it struggles with two major limitations in complex environments:

**Local Minima Traps:** The robot can get stuck in a "balancing act" between conflicting repulsive forces from multiple obstacles, failing to reach the goal.

Yeah, that is a very smart problem — like, as it tries to stay the right distance away from each thing without getting closer to another, it might never be able to move toward the target.

**Goal Not Reachable Near Obstacle (GNRON):** If a target point is placed right next to a wall or obstacle, the massive repulsive force can overpower the attractive force, making the robot unable to reach the destination.

And this too — it might not be able to reach the thing right next to it, so it has to go further away to even be able to approach. 

**How the Modified APF Solves These Issues**
To fix these flaws, engineers and researchers apply specific mathematical modifications to the potential functions:

**Adjusting the Repulsive Function:**
Instead of the traditional repulsive formula blowing up to infinity at an obstacle's edge, the modified equation incorporates the distance to the target into the repulsive calculation. This ensures that as the robot gets closer to the goal, the attractive force increases, allowing it to push past nearby obstacles.

That is very smart — if we factor distance to the attractive point into the equation, we can increase the attractive force as we get closer, so that even when obstacles are nearby, the robot has the confidence to power through.

**Dynamic Potential Fields:**
In some MAPF variations, the potential field is dynamically updated as the robot moves. If an obstacle is dynamic (moving), or if the robot enters an unexpected dead end, the algorithm reconstructs the potential field in real-time to find a viable path.

It's kind of funny that the field doesn't update as the robot keeps moving by default, but whatever — what they're essentially saying is that they're changing what counts as the attractive and repulsive points dynamically; they're trying to remap the environment on the fly.

**Virtual Subgoals & Path Smoothing:**
When the robot detects it is caught in a local minimum, the MAPF algorithm dynamically introduces virtual subgoals or "guiding vectors" that steer the robot sideways around the obstacle, allowing it to escape the trap and resume following the gradient toward the destination.

→ How does it know how to do this though? I'm not totally sure.

---

Yeah, I mean usually we'll slowly but surely train through all the different Q-values for different actions, whether it be tabular or DQN. But I understand their point — if there are a million actions you could take at a state, this might take too long to compute. It's easy if you can actually look up a table, but it's basically impossible when there are a huge number of actions at a single state, potentially infinite. So this is a way to solve that. And I feel like robotic arms and things like that definitely fall into this category.

---

Ok, so basically what I remember about this policy is that we check whether an action was advantageous or not using a Q-value or advantage calculation. This works either way since the rewards aren't what's being directly changed — something will output either Q or V. But once we figure out if an action was good or not, what did we do before? We increased the probability of that action being taken, and PPO clipped how much that increase could be. But what does this direct method do instead? It goes into the actor and makes it automatically decide, given a state input, what the output action should be — without a probability distribution in the middle. It looks at how the advantage or Q function changes with respect to a change in action, and adjusts the actor weights accordingly so that when a state is fed in, it either directly chooses that action or shifts toward it based on the gradient. Basically, this skips the probability step entirely.

---

Similarly, we might want our language model to be aware of a common misconception believed by 50% of people, but we certainly do not want the model to claim this misconception to be true in 50% of queries about it. In other words, selecting the model's desired responses and behavior from its very wide knowledge and abilities is crucial to building AI systems that are safe, performant, and controllable.

This is a very nice point that could support the idea that DPO with Plackett-Luce ranking could be done solely using LLM-generated feedback rather than relying on human feedback.