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

Tx_1D = PAMsymTx_Array.flatten() #make the arrays 1D and easier to use
Rx_1D = PAMsymRx_Array.flatten()

Tx_split = np.array_split(Tx_1D, 2)
Tx_train = Tx_split[0]
Tx_test = Tx_split[1]

Rx_split = np.array_split(Rx_1D, 2)
Rx_train = Rx_split[0]
Rx_test = Rx_split[1]


Rx_train_sm, _, Tx_train_sm, _ = train_test_split( #option of only using % of the dataset for faster testing
    Rx_train,Tx_train, test_size = 0.9
)

Rx_test_sm, _, Tx_test_sm, _ = train_test_split( #option of only using % of the dataset for faster testing
    Rx_test,Tx_test, test_size = 0.9
)

batch_size = 128
num_classes = 2
epochs = 30

model = Sequential()

model.add(Dense(1, activation='sigmoid'))
model.add(Dropout(0.1))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(5, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(num_classes, activation='softmax'))

#model.summary()

model.compile(loss='mse', optimizer=RMSprop(), metrics=['accuracy'])
history = model.fit(Rx_train_sm, Tx_train_sm,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(Rx_test_sm, Tx_test_sm))
score = model.evaluate(Rx_test_sm,Tx_test_sm, verbose=0)
print('Test loss:', score[0])
print('Test Accuracy:', score[1])


