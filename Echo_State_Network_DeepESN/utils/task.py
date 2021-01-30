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


def ber(toCheck, perfect):
    error = 0
    for i in range(perfect.size):
        if toCheck[i] != perfect[i]:
            error = error +1
    ber = error/perfect.size
    return ber

def roundTo(array):
    for i in range(array.size):
        if array[i] > 0:
            array[i] = 1
        else:
            array[i] = -1
    return array

        
def load_UWOC(path, metric_function):
    
    #data = loadmat(os.path.join(path, 'MG.mat')) # load dataset
    smallData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   
    largeData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps.mat')   
    dataset = Struct()    

    dataset.name = "PAMsym600mbps"
    Rx = smallData['PAMsymRx'].reshape((1,-1))#Mat']#.reshape((1,-1))
    Tx = smallData['PAMsymTx'].reshape((1,-1))#Mat']#.reshape((1,-1))
    print("Rx shape",Rx.shape)

    dataset.inputs = Rx[0:1004020]#15060300]
    dataset.targets = Tx[0:1004020]#15060300]
    print(dataset.inputs)
    print(dataset.targets)

    dataset.inputs = np.expand_dims(dataset.inputs, axis=0)
    #dataset.inputs = np.expand_dims(dataset.inputs, axis=0)
    dataset.targets = np.expand_dims(dataset.targets, axis=0)
    #dataset.targets = np.expand_dims(dataset.targets, axis=0)
    roundedin = roundTo(dataset.inputs.flatten())
    bitthing = ber(roundedin, dataset.targets.flatten())
    print("No equalised Ber: ", bitthing)
    print("Name",dataset.name)
    #print("Targets",dataset.targets)

    # input dimension
    Nu = dataset.inputs[0].shape[0]

    # function used for model evaluation
    error_function = metric_function    
    
    # select the model that achieves the maximum accuracy on validation set
    optimization_problem = np.argmin   
    
    
    #TR_indexes = range(150000) # indexes for training, validation and test set in Piano-midi.de task
    #VL_indexes = range(150000,251004)
    #TS_indexes = range(251004,502008)
    
    TR_indexes = range(300000) # indexes for training, validation and test set in Piano-midi.de task
    VL_indexes = range(300000,340000)
    TS_indexes = range(340000,400000)
    return dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes
  