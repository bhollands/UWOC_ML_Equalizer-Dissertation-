import tensorflow_probability as tfp  # We are using a different module from tensorflow this time
import tensorflow as tf

tfd = tfp.distributions  # making a shortcut for later on
initial_distribution = tfd.Categorical(probs=[0.2, 0.8])  # Refer to point 2 above
transition_distribution = tfd.Categorical(probs=[[0.5, 0.5],
                                                 [0.2, 0.8]])  # refer to points 3 and 4 above
observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])  # refer to point 5 above

# the loc argument represents the mean and the scale is the standard devitation

model = tfd.HiddenMarkovModel(
    initial_distribution=initial_distribution,
    transition_distribution=transition_distribution,
    observation_distribution=observation_distribution,
    num_steps=7)

mean = model.mean()
#due to the way TensorF;pw works omn a lower level we need to evaluate part of the graph
#from within a session to see the value of this tensor

#in the new versio of tf we need to use tf.compat.v1.Session() rather than just ft.Session()

with tf.compat.v1.Session() as sess:
    print(mean.numpy())