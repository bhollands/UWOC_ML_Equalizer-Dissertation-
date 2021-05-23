import numpy as np
from scipy.io import loadmat
from DeepESN import DeepESN
from utils import select_indexes
from utils import MSE
import os
import functools
import time 
import pandas as pd

def loadDeepESN(data):
    data = np.concatenate(data, 1)
    savedNetwork = np.genfromtxt('Wout.csv', delimiter = ",")
    savedNetwork = savedNetwork.reshape(1,-1)
    return savedNetwork[:,0:-1].dot(data) + np.expand_dims(savedNetwork[:,-1],1)


def loadData():

    smallData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')           
    Rx = smallData['PAMsymRx'].reshape((1,-1))#Mat']#.reshape((1,-1))
    Tx = smallData['PAMsymTx'].reshape((1,-1))#Mat']#.reshape((1,-1))
    #Rx = Rx[20000]
    #Tx = Tx[20000]
    Rx = np.expand_dims(Rx, axis=0)
    Tx = np.expand_dims(Tx, axis=0)
    #Tx = Tx.reshape(10,-1)
    return (Rx, Tx)
    

def loadState():
    chunk = pd.read_csv('states_2.csv', chunksize=1000)
    pd_df = pd.concat(chunk)
    pd_df.to_numpy()
    #states = np.genfromtxt('states.csv', delimiter = ",")
    return states.reshape(1,1,-1)





indexes = range(0,1004020)
Rx,Tx = loadData()

print("Loading States..." )
states = loadState()
print("Finished Lodaing")
start_time = time.time()
test_states = select_indexes(states, indexes)
result = loadDeepESN(test_states)
print("--- %s seconds ---" % (time.time() - start_time))

print(result.shape)
print(MSE(Tx, result))
print(result)


    
