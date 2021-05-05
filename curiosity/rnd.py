"""
Random Network Distillation

What's the core idea?
- 2 networks with different weights.
- One is a static network where weights don't change
- The other network tries to predict the output of the target model.
- reward is the prediction error for the next state's encodings

Pseudocode:

p, q = model(state1)
model1_state2_code = predictor_model(state2)
model2_state2_code = target_model(state2)

loss1 = ppo loss
loss2 = mse(model1_state2_code, model2_state2_code)

Experiment:
predict model2_state2_code from predictor_model(state1)

"""
