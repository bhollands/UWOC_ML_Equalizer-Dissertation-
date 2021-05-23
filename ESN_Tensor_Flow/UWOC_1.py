import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
from DeepRC import SimpleDeepESNClassifier

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

Rx = myData['PAMsymRx']
Tx = myData['PAMsymTx']

PAMsymRx_Array = Rx[0:1004020] #set to 1004040 for full dataset
PAMsymTx_Array = Tx[0:1004020]

scalar = MinMaxScaler(feature_range=(0,1))
datasetRx = scalar.fit_transform(PAMsymRx_Array)
datasetTx = scalar.fit_transform(PAMsymTx_Array)
(X_train, X_test, y_train, y_test) = train_test_split(
    datasetRx, datasetTx, test_size = 0.15, random_state=42
) #75% for training 25% for testing


model = SimpleDeepESNClassifier(num_classes = 2)

output = model.call(y_train)
plt.plot(output)
plt.xlim(0,50)
plt.show()



#model.compile(loss='mean_squared_error')

#model.fit(X_train, y_train)
# y_test_hat = ESN.predict(X_test)
# Train_MSE = ESN.MSE_Score(X_train, y_train)
# Test_MSE = ESN.MSE_Score(X_test, y_test)