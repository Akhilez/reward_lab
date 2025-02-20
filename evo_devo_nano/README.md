# Evo Devo Nano

What I want to do in this project:


- Simulate an animal-like learning behaviour. Be it evolutionary learning, developmental or neuronal learning.
- Incorporation of system 2 thinking -- 
  - the transformer model will have some "thinking" tokens 
  - that it can use to debrief and plan the next move.
- Overfitting and increasing complexity of the model.
- A gene pool and evolutionary learning process.
  - This can be done by considering a transformer layer as a gene.
  - Since the model is a residual stream with few layers adding their own information to the stream,
  - the model can be seen as a sequence of genes.
  - We can have a large pool of these transformer layers and select them based on some evolutionary fitness.
- Development could be like adding a new layer to the model in between
  - This new layer might have a small weight in the residual connection.
  - Slowly it might become more important.
- Will these layers be categorized as "early layers", "mid layers" and "late layers"?
- Will these be trained and persisted in the gene pool?
  - The developmental learning shouldn't really affect the gene pool, right?
- Maybe we can have statistics of which layers are more fit than others.
- Based on the fitness, we can decide which layers to keep and which to discard.
- If a layer does well, we can split it into two that will evolve separately.
- The model learns at every step. It has some limited short term memory part of the context.
- The long term memory will be the model weights themselves as it learns.
- Also, I'd like to add a memory module via some memory tokens that we can cross-attend to.
  - But idk how to backprop for long sequences.
- All learning schemes like predicting next action, value, reward, next state, inverse dynamics, 
  - should be done by the same model.
  - This is why we choose transformers.
  - Predicting the next state is a tough one though.


Env:
- The environment I'm thinking of is Craftax.
- We will ignore all the extrinsic rewards and just focus on the intrinsic ones.
- The human mind is so curious that we do crazy things without some God teaching us what to do.
- Similarly, the model will see the world online without replay buffer or in batches.
- And the model will learn through interaction with the environment.
- There are already some established methods that we can use.
  - Predicting next state given the current state and action.
  - Predicting action given the current state and next state.
  - Predicting actor's location given the current state and action.
- There should be some sort of sense of time for the agent. It should know how long it has been living.
- Some sort of spatial awareness via place cells/fields or grid cells or something.
- 

Possible training objectives:


Just obs and memory:
- [state tokens; action token]
  - Standard PPO with intrinsic reward
- [state tokens, action token; intrinsic reward]
- [state tokens, action token; next state tokens]
- [state1; action1, state2, reward1]
- CNN autoencoder state image to latent state to state image
- [state1] bert style

- [state tokens, action token, next state tokens; intrinsic reward]
  - intrinsic reward = error b/w predicted and actual next state tokens
- [state tokens, next state tokens; action token]
- [next state tokens, action token; state tokens]
- [state1; action1, state2, action2, state3, action3, ...]
- [state1; action1, reward1, state2, action2, reward2, state3, action3, reward3 ...]
- [state1; action1, state2, reward1, action2, state3, reward2, action3, state4, reward3, ...]
- [state1, state2; action1, state3, action2, state4, action3, ...]
- [state1, state2; action1, reward1, state3, action2, reward2, state4, action3, reward3, ...]
- [state1's "next_state" prefix output token, state2's "next_state" prefix output token; action1]  # inverse dynamics
- for all the above, use both state tokens and state's prefix output tokens