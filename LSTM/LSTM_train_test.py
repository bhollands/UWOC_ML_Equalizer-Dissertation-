'''
File for training and testing different Multi-Layer Perceptron networks for Equalisation

On all data set tried and amaounts avg loss = 0.68
https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
'''
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Activation

'''
Pre-processing
'''
myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

Rx = myData['PAMsymRx']#.reshape((1,-1))
Tx = myData['PAMsymTx']#.reshape((1,-1))

PAMsymRx_Array = Rx[0:20000] #set to 1004040 for full dataset
PAMsymTx_Array = Tx[0:20000]

# scalar = MinMaxScaler(feature_range=(0,1))
# PAMsymRx_Array = scalar.fit_transform(PAMsymRx_Array)
# PAMsymTx_Array = scalar.fit_transform(PAMsymTx_Array)

PAMsymRx_Array = np.reshape(PAMsymRx_Array, (PAMsymRx_Array.shape[0], 1, PAMsymRx_Array.shape[1]))
PAMsymTx_Array = np.reshape(PAMsymTx_Array, (PAMsymTx_Array.shape[0], 1, PAMsymTx_Array.shape[1]))



(X_train, X_test, y_train, y_test) = train_test_split(
    PAMsymRx_Array, PAMsymTx_Array, test_size = 0.15, random_state=42
) 


model = Sequential()
model.add(LSTM(4, input_shape=(1,1)))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=1)

#test_loss, test_accuracy = model.evaluate(X_test,y_test, verbose=0)

y_pred = model.predict(X_test) #use test set to find accuracy




plt.plot(y_test.flatten(), 'b-o', markersize = 4, label="Ideal ")
plt.plot(y_pred, 'r-o', markersize = 4, label="Output")
#plt.plot(test_states, 'g-o', markersize = 4, label="Input")
plt.legend(loc='upper right')
plt.xlabel("Number of bit")
plt.ylabel("Bit status")
plt.xlim([0,50])
plt.show()


# plt.plot(y_pred, label="prediction")
# plt.plot(X_test.flatten(), label="input")
# plt.plot(y_test.flatten(), label="Actual")

# plt.legend(loc='upper right')
# plt.xlim([0,100])
# plt.show()
# #saveResults(y_pred, X_test, y_test)
# print('Test loss:', test_loss)
# print('Test Accuracy:', test_accuracy)

# '''
# plt.plot(PAMsymRx_Array)
# plt.plot(PAMsymTx_Array)
# plt.xlim([0,100])
# plt.show()
# '''

# '''
# Saving Results
# '''
# def saveResults(predict,input_values, perfect):
#     newArr = np.column_stack((predict, input_values, perfect))
#     predict = pd.DataFrame(newArr)

#     output_filepath = 'LSTM\Results\_results_keras.xlsx'
#     predict.to_excel(output_filepath, index = False)
# #print('Test loss:', test_loss)
# #print('Test Accuracy:', test_accuracy)


