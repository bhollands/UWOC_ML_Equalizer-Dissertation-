import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#need to make train set and test set

in_file = os.path.dirname(os.path.abspath(__file__)) #fix later
Rx_datapath = os.path.join(in_file, 'PAMsymRx.csv')
Tx_datapath = os.path.join(in_file, 'PAMsymTx.csv')


PAMsymTx = pd.read_csv(Tx_datapath)# read in csv data 
PAMsymRx = pd.read_csv(Rx_datapath)

PAMsymTx_Array = PAMsymTx.values #turn value into usable numpy array
PAMsymRx_Array = PAMsymRx.values


(X_train, X_test, y_train, y_test) = train_test_split(
    PAMsymRx_Array, PAMsymTx_Array, test_size = 0.15, random_state=42
) #75% for training 25% for testing


batch_size = 128
num_classes = 1
epochs = 5

model = Sequential()

model.add(Dense(1, activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(5, activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(5, activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='tanh'))

#model.summary()

model.compile(loss='mse', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test))
score = model.evaluate(X_test,y_test, verbose=0)
print('Test loss:', score[0])
print('Test Accuracy:', score[1])


