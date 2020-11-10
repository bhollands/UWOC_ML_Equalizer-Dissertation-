import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
import tensorflow as tf
from tensorflow import keras


in_file = os.path.dirname(os.path.abspath(__file__)) #fix later
Rx_datapath = os.path.join(in_file, 'PAMsymRx.csv')
Tx_datapath = os.path.join(in_file, 'PAMsymTx.csv')


PAMsymTx = pd.read_csv(Tx_datapath)# read in csv data 
PAMsymRx = pd.read_csv(Rx_datapath)

PAMsymTx_Array = PAMsymTx.values #turn value into usable numpy array
PAMsymRx_Array = PAMsymRx.values

PAMsymTx_Array.flatten() #make the arrays 1D and easier to use
PAMsymRx_Array.flatten()


PAMsymRx_Array, _, PAMsymTx, _ = train_test_split( #option of only using % of the dataset for faster testing
    RX_sm, TX_sm,  test_size = 0.9
)


model = keras.Sequential ([
    keras.layers.Dense(1)
    keras,layers.Dense(128, activation='relu')
    keras,layers.Dense(2, activation='relu')
])

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_Rx, train_Tx, epochs=10)
test_loss, test_acc = model.evaluate(test_Rx, Test_Tx, verbose=0)
print('Test accuracy', test_acc)
