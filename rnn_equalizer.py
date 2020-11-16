import pandas as pd
import numpy as np
import os
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
# fix random seed for reproducibility
np.random.seed(7)

#load the dataset but only keep the top n words, zero the rest
#top_words = 5000
#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


in_file = os.path.dirname(os.path.abspath(__file__)) #fix later
Rx_datapath = os.path.join(in_file, 'PAMsymRx.csv')
Tx_datapath = os.path.join(in_file, 'PAMsymTx.csv')


PAMsymTx = pd.read_csv(Tx_datapath)# read in csv data 
PAMsymRx = pd.read_csv(Rx_datapath)

PAMsymTx_Array = PAMsymTx.values #turn value into usable numpy array
PAMsymRx_Array = PAMsymRx.values

Tx_1D = PAMsymTx_Array.flatten() #make the arrays 1D and easier to use
Rx_1D = PAMsymRx_Array.flatten()

Tx_split = np.array_split(Tx_1D, 2)
Tx_train = Tx_split[0]
Tx_test = Tx_split[1]

Rx_split = np.array_split(Rx_1D, 2)
Rx_train = Rx_split[0]
Rx_test = Rx_split[1]


(X_train, X_test, y_train, y_test) = train_test_split(
    Rx_1D, Tx_1D, test_size = 0.25, random_state=42
) #75% for training 25% for testing



#max_review_length = 500
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

no_input_dim = X_train.size
print(X_train)

model = Sequential()
model.add(Embedding(input_dim=no_input_dim, output_dim=64))

model.add(LSTM(128)) # LSTM with 100 internal units
model.add(Dense(2, activation='softmax'))
print(model.summary())


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3,verbose=1, batch_size=64)

 	
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
