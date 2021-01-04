'''
File for training and testing different Multi-Layer Perceptron networks for Equalisation

On all data set tried and amaounts avg loss = 0.68
'''

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

Rx = myData['PAMsymRx']#.reshape((1,-1))
Tx = myData['PAMsymTx']#.reshape((1,-1))

PAMsymRx_Array = Rx[0:1004020] #set to 1004040 for full dataset
PAMsymTx_Array = Tx[0:1004020]

scalar = MinMaxScaler(feature_range=(0,1))
datasetRx = scalar.fit_transform(PAMsymRx_Array)
datasetTx = scalar.fit_transform(PAMsymTx_Array)
(X_train, X_test, y_train, y_test) = train_test_split(
    datasetRx, datasetTx, test_size = 0.15, random_state=42
) #75% for training 25% for testing


batch_size = 64
epochs = 5
model = Sequential()

model.add(Dense(1, activation='tanh')) #first layer
model.add(Dense(16, activation='tanh')) #hidden layers
model.add(Dense(16, activation='tanh'))
model.add(Dense(16, activation='tanh'))
model.add(Dense(1, activation='tanh')) #output layer

opt = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD") #gradient decent

model.compile(loss='mse', optimizer='SGD', metrics=['accuracy']) #compile the model

checkpoint_path = "MLP\model_checkpoints\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#create checkpoint callback
cp_callback =tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)
 #training
history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks = [cp_callback])

test_loss, test_accuracy = model.evaluate(X_test,y_test, verbose=0)

y_pred = model.predict(X_test, batch_size=64, verbose=0) #use test set to find accuracy
#print(y_pred)

def saveResults(predict,input_values, perfect):
    newArr = np.column_stack((predict, input_values, perfect))
    predict = pd.DataFrame(newArr)

    output_filepath = 'MLP\Results\_results_keras.xlsx'
    predict.to_excel(output_filepath, index = False)
  


saveResults(y_pred, X_test, y_test)

print('Test loss:', test_loss)
print('Test Accuracy:', test_accuracy)


