import tensorflow as tf
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
from model import EchoStateNetwork

import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler

myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

Tx = myData['PAMsymTx']


data = Tx[0:1004020].flatten()


scalar = MinMaxScaler(feature_range=(0,1))

#datasetTx = scalar.fit_transform(PAMsymTx_Array)
# (X_train, X_test, y_train, y_test) = train_test_split(
#     datasetRx, datasetTx, test_size = 0.15, random_state=42
# ) #75% for training 25% for testing

#create data
#predict the next 1 observation from the subsequense of length d
window_size_d = 16
pred_length_l = 1

def create_sliding_window_data(X, window_size, pred_length):
    data = [X[i:i+window_size] for i in range(len(X)-window_size-pred_length)]
    target = [X[i+window_size:i+window_size+pred_length] for i in range(len(X)-window_size-pred_length)]
    return np.array(data,dtype=np.float32), np.array(target,dtype=np.float32)

X_windows, y_windows = create_sliding_window_data(data,window_size_d,pred_length_l)
X_train, X_test, y_train, y_test = train_test_split(X_windows, y_windows, test_size=0.8,shuffle=False)
print(X_train)


X_train = np.expand_dims(X_train,axis=2)
X_test = np.expand_dims(X_test,axis=2)

#run ESN
ESN = EchoStateNetwork(units=32)
ESN.fit(X_train, y_train)
print("Train MSE:", ESN.MSE_Score(X_train, y_train))
print("Test MSE:", ESN.MSE_Score(X_test, y_test))
y_test_hat = ESN.predict(X_test)

#Result
tr_size = len(X_train)
plt.figure(figsize=(16, 8))
plt.xlim(300000, 300100)
plt.plot(data, linewidth = 2, label='data')
plt.plot(range(window_size_d+tr_size, window_size_d+tr_size+len(y_test_hat)), y_test_hat, linewidth = 2, alpha=0.9, label='predict')
plt.axvspan(0, tr_size+window_size_d, facecolor='g', alpha=0.1, label =  "train area")
plt.legend(loc='upper right')
plt.show()