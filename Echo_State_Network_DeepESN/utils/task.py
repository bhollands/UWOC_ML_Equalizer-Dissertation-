'''
Task configuration file
    
----

This file is a part of the DeepESN Python Library (DeepESNpy)

Luca Pedrelli
luca.pedrelli@di.unipi.it
lucapedrelli@gmail.com

Department of Computer Science - University of Pisa (Italy)
Computational Intelligence & Machine Learning (CIML) Group
http://www.di.unipi.it/groups/ciml/

----
'''

import functools
import os
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class Struct(object): pass

def select_indexes(data, indexes, transient=0):

    if len(data) == 1:
        return [data[0][:,indexes][:,transient:]]
    
    return [data[i][:,transient:] for i in indexes]

def load_pianomidi(path, metric_function):
    
    data = loadmat(os.path.join(path, 'pianomidi.mat')) # load dataset

    dataset = Struct()
    dataset.name = data['dataset'][0][0][0][0]
    dataset.inputs = data['dataset'][0][0][1][0].tolist()
    dataset.targets = data['dataset'][0][0][2][0].tolist()

    #print("input", dataset.inputs)
    #print("targets", dataset.targets)
    # input dimension
    Nu = dataset.inputs[0].shape[0]

    # function used for model evaluation
    error_function = functools.partial(metric_function, 0.5)     
    
    # select the model that achieves the maximum accuracy on validation set
    optimization_problem = np.argmax    
    
    
    TR_indexes = range(87) # indexes for training, validation and test set in Piano-midi.de task
    VL_indexes = range(87,99)
    TS_indexes = range(99,124)
    
    return dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes


def load_UWOC(path, metric_function):
    
    #data = loadmat(os.path.join(path, 'MG.mat')) # load dataset
    '''
    smallData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   
    largeData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps.mat') 
    '''
    #data_set_name = 'POF60m_PAMExp_2PAM_DR600Mbps(single column).mat'
    #data_set_name = 'UWOC27m_PAM16_250Mb_APDGain100_P13dBm.mat'
    #data_set_name = 'UWOC27m_PAM16_125Mb_APDGain100_P13dBm.mat'
    #data_set_name = 'UWOC27m_PAM4_125Mb_APDGain80_P7dBm.mat'
    #data_set_name = 'UWOC27m_PAM8_94Mb_APDGain100_P10dBm.mat'
    data_set_name = 'UWOC27m_PAM8_188Mb_APDGain100_P10dBm.mat'
    
    data = loadmat(data_set_name)

    dataset = Struct()    

    dataset.name = "PAMsym600mbps"
    Rx = data['PAMsymRx']#.reshape((1,-1))#Mat']#.reshape((1,-1))
    Tx = data['PAMsymTx']#.reshape((1,-1))#Mat']#.reshape((1,-1))

    dataset = Struct() 

    Rx = np.expand_dims(Rx, axis=0)
    Rx = Rx.reshape(-1,1)


    Tx = np.expand_dims(Tx, axis=0)
    Tx = Tx.reshape(-1, 1)
    #print("Rx shape",Rx.shape)
    scaler = MinMaxScaler(feature_range=(0,1))
    
  
    #print("Before Scaling", Rx)
    scale = True
    if scale:
        Rx = scaler.fit_transform(Rx).reshape((1,-1))
        Tx = scaler.fit_transform(Tx).reshape((1,-1))
        dataset.inputs = Rx[0:240000]#15060300] #now have 4 either side
        dataset.targets = Tx[0:240000]#.flatten()#15060300]
    else:
        dataset.inputs = Rx[0:240000].reshape((1,-1))#15060300]
        dataset.targets = Tx[0:240000].reshape((1,-1))#.flatten()#15060300]

    dataset.inputs = np.expand_dims(dataset.inputs, axis=0)
    dataset.targets = np.expand_dims(dataset.targets, axis=0)
    
    # input dimension
    Nu = dataset.inputs[0].shape[0]
    print('Nu: ', Nu)

    # function used for model evaluation
    error_function = metric_function    
    
    # select the model that achieves the maximum accuracy on validation set
    optimization_problem = np.argmin   
    

    TR = 15000
    VL = TR+5000
    TR_indexes = range(TR) # indexes for training, validation and test set in Piano-midi.de task
    VL_indexes = range(TR, VL)
    TS_indexes = range(VL,240000)

    return dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes, scaler, data_set_name
  