#https://www.tensorflow.org/guide/keras/rnn

import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


'''
Here is a simple example of a Sequential model that processes sequences of integers, embeds each integer into a 64-dimensional vector, then processes the sequence of vectors using a LSTM layer.
'''

model = keras.Sequential()
# Add an Embedding layer expecting input vocab of size 1000, and
# output embedding dimension of size 64.
model.add(layers.Embedding(input_dim=1000, output_dim=64))

#Add a LSTM layer with 128 internal units
model.add(layers.LSTM(128))

#Add a dense layer with 10 units .
model.add(layers.Dense(10))

#model.summary()

'''
In addition, a RNN layer can return its final internal state(s). The returned states can be used to resume the RNN execution later, or to initialize another RNN. 
This setting is commonly used in the encoder-decoder sequence-to-sequence model, where the encoder final state is used as the initial state of the decoder.

To configure a RNN layer to return its internal state, set the return_state parameter to True when creating the layer. Note that LSTM has 2 state tensors, but GRU only has one.

To configure the initial state of the layer, just call the layer with additional keyword argument initial_state. 
Note that the shape of the state needs to match the unit size of the layer, like in the example below.
'''

encoder_vocab = 1000
decoder_vocab = 2000

encoder_input = layers.Input(shape=(None,))
encoder_embedded = layers.Embedding(input_dim=encoder_vocab, output_dim=64)(
    encoder_input
)

#return states in addition to output
output, state_h, state_c = layers.LSTM(64, return_state=True, name="encoder")(
    encoder_embedded
)
encoder_state = [state_h, state_c]

decoder_input = layers.Input(shape=(None,))
decoder_embbeded = layers.Embedding(input_dim=decoder_vocab, output_dim=64)(
    decoder_input
)

#pass the 2 states to a new LSTM layer, as initial state
decoder_output = layers.LSTM(64, name='decoder')(
    decoder_embbeded, initial_state=encoder_state
)
output = layers.Dense(10)(decoder_output)

model = keras.Model([encoder_input, decoder_input], output)
#model.summary()

#Cross-batch statefulness
'''
When processing very long sequences (possibly infinite), you may want to use the pattern of cross-batch statefulness.

Normally, the internal state of a RNN layer is reset every time it sees a new batch (i.e. every sample seen by the layer is assumed to be independent of the past).
The layer will only maintain a state while processing a given sample.

If you have very long sequences though, it is useful to break them into shorter sequences, 
and to feed these shorter sequences sequentially into a RNN layer without resetting the layer's state. That way, the layer can retain information about the entirety of the sequence, even though it's only seeing one sub-sequence at a time.

You can do this by setting stateful=True in the constructor.

If you have a sequence s = [t0, t1, ... t1546, t1547], you would split it into e.g.

s1 = [t0, t1, ... t100]
s2 = [t101, ... t201]
...
s16 = [t1501, ... t1547]

then you would precess it via
'''

#lstm_layer = layers.LSTM(64, stateful=True)
#for s in sub_sequences:
#    output = lstm_layer(s)

'''
a complete example of layer.reset_states()
'''
paragraph1 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph2 = np.random.random((20, 10, 50)).astype(np.float32)
paragraph3 = np.random.random((20, 10, 50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)
output = lstm_layer(paragraph3)

# reset_states() will reset the cached state to the original initial_state.
# If no initial_state was provided, zero-states will be used by default.
lstm_layer.reset_states()

'''
RNN State Resue
The recorded states of the RNN layer are not included in the layer.weights(). 
If you would like to reuse the state from a RNN layer, you can retrieve the states value by layer.states and use it as the initial state for a new layer
via the Keras functional API like new_layer(inputs, initial_state=layer.states), or model subclassing.

Please also note that sequential model might not be used in this case since it only supports layers with single input and output, 
the extra input of initial state makes it impossible to use here.
'''

paragraph1 = np.random.random((20,10,50)).astype(np.float32)
paragraph2 = np.random.random((20,10,50)).astype(np.float32)
paragraph3 = np.random.random((20,10,50)).astype(np.float32)

lstm_layer = layers.LSTM(64, stateful=True)
output = lstm_layer(paragraph1)
output = lstm_layer(paragraph2)

existing_state = lstm_layer.states

new_lstm_layer = layers.LSTM(64)
new_output = new_lstm_layer(paragraph3, initial_state=existing_state)


'''
Bidirectional RNNs

For sequences other than time series (e.g. text), it is often the case that a RNN model can perform better if it not only processes sequence from start to end, but also backwards. 
For example, to predict the next word in a sentence, it is often useful to have the context around the word, not only just the words that come before it.

Keras provides an easy API for you to build such bidirectional RNNs: the keras.layers.Bidirectional wrapper.
'''

model = keras.Sequential()

model.add(
    layers.Bidirectional(layers.LSTM(64, return_sequences=True), input_shape=(5,10))
)

model.add(layers.Bidirectional(layers.LSTM(32)))
model.add(layers.Dense(10))

model.summary()

#get GPU