'''
Script below is for loading trained model and passing new data through it
'''

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
import tensorflow as tf
from tensorflow import keras
from scipy.io import loadmat
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop


in_file = os.path.dirname(os.path.abspath(__file__)) #fix later

checkpoint_path = "MLP\model_checkpoints\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps.mat')   

Rx = myData['PAMsymRxMat'].flatten()#.reshape((1,-1))
Tx = myData['PAMsymTxMat'].flatten()#.reshape((1,-1))

PAMsymRx_Array = Rx[1000000:2000000] #set to 15060300 for full dataset
PAMsymTx_Array = Tx[1000000:2000000]

def saveResults(predicted, input_values, perfect):
    newArr = np.column_stack((predicted,input_values, perfect))
    predict = pd.DataFrame(newArr)

    output_filepath = 'MLP\Results\_results_keras_trained_model.xlsx'
    #fitness_filepath = 'NN_output data\_fitness_results_'+file +'_'+activeFunc+'.xlsx'
    predict.to_excel(output_filepath, index = False)
    #fitness.to_excel(fitness_filepath, index = False)

def create_model():
    model = Sequential()
    model.add(Dense(1, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    model.add(Dense(16, activation='tanh'))
    
    model.add(Dense(1, activation='tanh'))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    return model


model = create_model()
model.load_weights(checkpoint_path)
loss, acc = model.evaluate(PAMsymRx_Array,PAMsymTx_Array)
print("Trained model:", loss)
y_pred = model.predict(PAMsymRx_Array, batch_size=128, verbose=0)
#saveResults(y_pred,PAMsymRx_Array, PAMsymTx_Array)
