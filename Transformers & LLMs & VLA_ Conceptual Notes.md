# Transformers & LLMs: Conceptual Notes

---

## 1. Word Embeddings — How Words Become Vectors

A word can be mapped into a vector. Since we have training data where we know what the output should be for a given piece of text, we can retrofit that back through backprop. And even though you might ask "how is the first input adjusted — I thought we only change weights and biases during backprop?" — the special thing is that we also know how each activation number should change in relation to the weight and bias being changed, which is what allows that backward motion to propagate all the way through. So even when we get to the very first step, we will have suggested nudges on those initial activation numbers as well.

![image](https://hackmd.io/_uploads/BJdWbtEZGx.png)

This is essentially the first input. And the funny thing is, through training, that embedding matrix can and should very much be learned — there's no need for a hardcoded lookup. The backprop nudges the word vectors themselves over time.

One beautiful result of this: if you go one direction in embedding space, that direction might encode plurality — like "cat" vs "cats." Go in another direction and it might encode gender. So if you take the difference between "mom" and "dad" and add it to "aunt," you might get close to the encoding of "uncle." The directions in the space *mean something*, even though we never explicitly told it to work that way.

And just to be clear on dot products: the dot product between two vectors tells us how close or not close two things are — it sums up all the components multiplied together. But the key thing to keep in mind is that in a deep network, we don't always know the *purpose* of what we're checking is similar or different. We're just putting in a weight matrix to convert from one thing to the next. It's not like a photo kernel where I can directly say "this is what we're checking is equal or different." The network figures that out.

---

## 2. Attention — How Words Influence Each Other

The reason a word might encode another word as part of attention is that — for example, the word "bank" — if the words "water" or "river" are nearby, attention allows it to encode the river meaning into "bank" during that process. That's the beauty of it.

HAHA I really like how they called it "attention" now. That making of one column having info about the other columns near it — it's like a projection or an addition of two vectors, but the key thing is that even the further-away ones can impact it, *but only up to a certain number of words*. So calling it attention, for that reason, is actually very nice.



### Queries, Keys, and Values

A query asks an all-encompassing question suitable for the word. A key gives an answer. And if that dot product between the query and key is high, it signals that the query got an answer — meaning those two vectors are near each other.

For example, the word "creature" might ask: *"are there any adjectives near me?"* The word "fluffy" might have a key that signals *"I'm an adjective."* When the dot product between creature's query and fluffy's key is high, we know the question was answered.
![image](https://hackmd.io/_uploads/SyBUXi4Zze.png)


But the answer isn't just yes or no — it's a *value vector*. The value vector is what we add to the word to encode that meaning. Like "gender" could be a direction that gets added to make something something else. Those weights allow the model to adjust to that.

And here's why scaling by the dot product makes sense: the dot product tells us just how much impact that word should have. So if we scale the value with that number, we know not to add *all* of that direction but just the meaningful amount. And if we add all of those scaled value vectors together first before adding to the main word — that's doing the exact same thing as adding the value vectors one at a time. It's the same result, just more efficient.

So to sum it up: for each word, depending on how much each other word's key matched its query, we add up those weighted value vectors, then add the sum to the original word. That's the attention update. And this gives us our new, context-enriched list of word representations — the attention output.


![image](https://hackmd.io/_uploads/B1SD8XUZMe.png)


### The Size Constraint

One thing important to notice: it doesn't matter how many activation numbers were in the previous layer. We just need the size of the weight matrix's rows to match the size of the activation numbers so the dot product works. What matters in our case is the number of rows — which determines how many output neurons we get.

For the final layer, since we want probabilities over 50,000 words, we need our output to be 50,000 long — one neuron per word, with its activation number representing the probability of that word being next. The unembedding matrix does exactly that.

![image](https://hackmd.io/_uploads/rJbIBtNZGe.png)

### Masking During Training

We only allow a token to attend to tokens that come *before* it, not after — because we're simultaneously training the model to guess each next word in the sequence. Since we already know all the words in the training text, we can ask "can the model predict the word after token 1? After token 2? After token 3?" all at once. One training example becomes multiple training examples, which is why we guess all possible next words throughout the sequence rather than just the final one.

The masking is what makes this safe — attention of encoding meanings into a word for the next-word guess only applies forward, not backward, so we can train simultaneously across the whole sequence.
![image](https://hackmd.io/_uploads/rk98654Wze.png)
![image](https://hackmd.io/_uploads/S1h4uoV-Ge.png)


### A Note on Training Bias

One subtle issue: if AI-generated text is used as training data, we're retrofitting parameters to match what the AI already said. Imagine training on "hi my name is" — the model guesses "your" but the training text says "my." Now the model has to adjust away from a perfectly valid prediction. But since it's trained on many diverse examples, the model isn't going to say "hi pooping" or something — it evens out. Just worth keeping in mind.

### Self-Attention vs. Cross-Attention

- **Self-attention:** each token attends to other tokens in the *same* sequence
- **Cross-attention:** tokens from one sequence attend to tokens from a *different* sequence — used in translation (our language → another language), or in vision-language models where text tokens attend to image patches. "I do not want to pet it" will never insert context from itself into its words, which is why self attention is desirable even in translation.


![image](https://hackmd.io/_uploads/S10IRXIZMg.png)

### Multi-Head Attention

From a single attention head, we get suggested changes to each word.
one word change:![image](https://hackmd.io/_uploads/SkASQEUbGx.png)
multiple word change in ONE attention head:![image](https://hackmd.io/_uploads/H1DDv7Ibzg.png) If we do multiple heads, we get multiple sets of suggested changes. Each head uses different Q/K/V weight matrices, so each head can specialize in probing for different things — one might look for adjective-noun relationships, another for coreference, another for something else entirely.

For all of them, you get the sum of (value × key·query dot product) added to each word, computed once per head. Then we do that multiple times with different weights.


![image](https://hackmd.io/_uploads/HJ8cGE8Zzg.png)

For backprop through multi-head attention: if I know what aspects of E8 (the final word vector) should have been higher or lower, and I know how many additions were made to each part of it from the value updates, then I know exactly what to adjust. I go to each of those value, key, and query weights that contributed and nudge them accordingly. And the value vectors came from separate word conversions too, which also tells us how to change those original word vectors.



### What Each Part of the Attention Output Encodes

Each dimension of the attention output answers something. Maybe one dimension checks "is this English?" — just positive or negative. Maybe another checks "is this related to Harry Potter?" The keys map each token to answers for those kinds of questions, and the query checks if any of those answers fire. These aren't things we explicitly design — they emerge from training.

![image](https://hackmd.io/_uploads/rkZFZ7PWzg.png)

---

## 3. The Feed-Forward Network — Pattern Matching & Fact Encoding

This is also sometimes called the MLP (multi-layer perceptron) sublayer. The way to think about it:

**First layer:** checks "does this token have feature X?" — e.g., is "Michael Jordan" present? Is this English text?

**ReLU activation:** only fires if the answer clears a threshold (this is where bias comes in — bias isn't too deep, it's just saying "only activate if you're past this point," like subtracting 10 so the neuron only turns on if the result is above 10).

**Second layer:** if yes, adds back a specific direction to the word vector — encoding the associated fact. Like if Michael Jordan is present, add exactly this much in the "basketball" direction.

The Micheal Jordan example is great here. Best case, if we have both the "Michael" and "Jordan" vectors, we get a result of 2. With a bias of -1, that becomes 1 — positive, so ReLU activates. If only one was present, we'd get 1 - 1 = 0, which is neutral. If neither, negative, which ReLU kills to 0. So it becomes an AND gate — *both* Michael *and* Jordan have to be present for this to fire. And then the second layer adds the "basketball" meaning encoding. If Michael Jordan wasn't present at all, nothing gets added. That's exactly the right behavior.


![image](https://hackmd.io/_uploads/SJJprQwZMg.png)
![image](https://hackmd.io/_uploads/rygkZQDbzx.png)



The same framework applies more generally: the first weight layer checks if something is there, and the second weight layer tells us what to add to our original encoding because of it. We're going from our word encoding, up to a higher-dimensional space to do the checking, then back down to the original size to add the update.

### Superposition — Why More Dimensions Is So Powerful

Here's a beautiful insight. In a 3-dimensional space, you can only have up to 3 perfectly perpendicular (orthogonal) directions. But in a 10,000-dimensional space, you can have *way more than 10,000 nearly-orthogonal directions* — this is called superposition.

Why does this matter? Because the first layer of the feed-forward network is asking "does this feature exist?" using AND gates across rows. If you have more orthogonal directions available, you can use *more rows* to answer the same question.

For example, let's say we want to detect whether Michael Jordan is present. Instead of being forced to capture that with just one row (the "Michael + Jordan" gate), if we have more near-orthogonal directions, we can have multiple rows probing for the same phenomenon from different angles — "Michael + Jordan," "Michael + his jersey number," "Jordan + basketball," whatever. More rows means more gates helping confirm or deny the presence of a concept, which means we can encode more things reliably even in a fixed-size model.![image](https://hackmd.io/_uploads/HyqzWNwWMe.png)

So superposition allows the model to store far more "concepts" than it has raw dimensions, by packing them into slightly non-orthogonal directions. And the 89° and 91° directions — things that are *almost* perpendicular — can still encode a direction that gets checked meaningfully. This is why large embedding dimensions matter so much.

![image](https://hackmd.io/_uploads/ByWTzNPZGe.png)

---

## 4. Unembedding & Softmax — Getting to Word Probabilities

At the end of the transformer, we take the final token's vector and multiply it by an **unembedding matrix** to get a vector row of length 50,000 — one score per word in the vocabulary. Each neuron in that output represents a word, and its activation number is meant to represent the probability of that word being the next one.![image](https://hackmd.io/_uploads/HkDI7KNWGg.png)

Then we run it through **softmax** to convert those raw scores into something that sums to 1. The way softmax works: take e to the power of each score, add them all up, and divide each one by that sum. So each value becomes a fraction of the total — a probability-like distribution.

The guy in the video said it wasn't intuitive, but honestly it is. We're adding up everything and dividing, so we get a legit probability distribution.

![image](https://hackmd.io/_uploads/BJb7YKEbMg.png)
![image](https://hackmd.io/_uploads/BJU9YFEZMg.png)

### Temperature

We don't always want to pick the highest-probability word every time — that leads to repetitive output. Temperature lets us control how much randomness to introduce.

Here's the intuition: pretend temperature is a small decimal like 0.2. If we compute e^(high_value / 0.2), that's the same as e^(high_value × 5) — the exponent gets amplified enormously. So a word that was already high-probability gets *way* more weight relative to lower-probability words. The differences get exaggerated.

Flip it: if temperature is large (e.g., 1.5), all the exponents get shrunk. The gap between high and low scores compresses. More probability mass spreads to lower-probability words — more randomness, more creative output.

So: smaller temperature → higher-probability words dominate even more. Larger temperature → the distribution flattens, giving lower-probability words more of a shot.

![image](https://hackmd.io/_uploads/r1ibHqVbzg.png)

---

## 5. Backpropagation Through All of This

Backprop works the same way throughout the entire transformer — we track how each output depends on each parameter and nudge accordingly.

For the attention mechanism: if we know what aspects of a word vector should have been higher or lower based on the loss, and we know what value additions were made to it from each attention head, we can trace back to the Q, K, and V weight matrices and nudge them. We can also trace back to the original word encodings themselves — and this is what causes similar words to end up mapped near each other over training.

For the feed-forward: same logic. If the activation was wrong, nudge the weights that produced it.

For the embedding matrix: backprop gives us suggested nudges on the initial word vectors, which is what causes the geometry of the embedding space to become meaningful over time — words that appear in similar contexts drifting toward each other, and directions like gender or plurality emerging.



---

## 6. Vision-Language Models & Robot Action Models

### Image Tokens

Images can be turned into tokens too. A **projector** network takes image patches and converts them into vectors of the same size as word embeddings. Once they're the same size, self-attention (or cross-attention) can mix image tokens and text tokens together.

Maybe instead of self-attention, cross-attention is used here — but the key thing is the projection makes all tokens the same size so attention can happen across modalities.

![image](https://hackmd.io/_uploads/r1M9Yc_Wzx.png)
![image](https://hackmd.io/_uploads/ryJn89ObGx.png)

In the same way a query derived from "creature" might attend to nearby adjective tokens, a query derived from the word "pen" might have a high dot product with the key from the image patch that *contains the pen*. So the pen image region's value vector gets added to the word "pen" — encoding visual properties like color and location into that text token. And remember, it doesn't matter what we applied the query to — if the query and key are near each other, the value gets added. It works just as well across modalities.

![image](https://hackmd.io/_uploads/HJYzGk5Wfg.png)
![image](https://hackmd.io/_uploads/SkFoEk5Zze.png)

### Robot Action Models

Instead of predicting the next *word*, the final layer can be replaced to predict the next *action* — like joint angles or gripper commands. The unembedding step just becomes a different kind of output head.

Training requires paired data: instructions + camera images + proprioception (joint states, forces, gripper status) + action sequences. The model sees all of that and learns to output the right motor command.

![image](https://hackmd.io/_uploads/H1ZK4M9-zg.png)
![image](https://hackmd.io/_uploads/Sy31p0tbGx.png)
![image](https://hackmd.io/_uploads/Skln0RKZfl.png)

Some architectures predict trajectories across multiple future timesteps (e.g., 50 steps ahead) rather than just the immediate next action. A **diffusion model** is used for this: start with a random noisy trajectory and iteratively denoise it into a clean, physically plausible action sequence. The core idea of diffusion: the network learns to look at a noisy version of the data and predict exactly how much noise to remove, step by step.

One question about the architecture: you want to predict trajectories for the next 50 timesteps, but you need to be at a single timestep to do that. The LLM part can move across timesteps because the image we feed is across time. And for training the action model, we can train on the same image multiple times, which is nice. Probably the last step of the diffusion model changes in the sense that it tells you what batch of actions comes next, then it's like the LLM adding that on and going from there again.

**A huge architectural insight:** the LLM attention heads and the action model attention heads can run in *parallel* at the same time. We don't have to wait for one to finish before starting the other — we can compute each head simultaneously. That's what makes that shared architecture so powerful.

![image](https://hackmd.io/_uploads/SyhFUf5WMx.png)

**Key safety insight:** if the VLM glitches for a split second, a standalone action expert with its own local memory can keep the robot moving safely without suddenly freezing or crashing. This is an important reason for having a dedicated action model rather than having the VLM do everything.

Data structure for training a robot policy:
```
{
  instruction:    "pick up the red block",
  observations:   [camera images across time],
  proprioception: [joint angles, forces, gripper state],
  actions:        [sequence of motor commands]
}
```

---

## Summary of the Full Pipeline

```
Raw text / images
       ↓
Token embeddings (learned via backprop — similar words drift together over training)
       ↓
Positional encoding (so the model knows word order)
       ↓
[Repeated N times:]
  → Multi-head self-attention  (tokens absorb context from each other via Q/K/V)
  → Feed-forward network       (AND-gate feature detection + fact encoding)
       ↓
Final token vector
       ↓
Unembedding matrix → 50,000-dim logits (one per vocabulary word)
       ↓
Softmax (+ temperature) → probability distribution over next token
       ↓
Sample next token
```

For robot models, the last steps swap out vocabulary probabilities for action probabilities or trajectory predictions via a diffusion model.