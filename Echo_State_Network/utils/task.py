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

def load_UWOC(path, metric_function):
    
    #data = loadmat(os.path.join(path, 'MG.mat')) # load dataset
    myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

    dataset = Struct()
    #myStuff = Struct()
    #dataset.name = data['dataset'][0][0][0][0]
    #dataset.inputs = data['dataset'][0][0][1][0]
    #dataset.targets = data['dataset'][0][0][2][0]

    #print(dataset.targets)
    
   
    dataset.name = "PAMsym500mbps"
    Rx = myData['PAMsymRx'].reshape((1,-1))
    Tx = myData['PAMsymTx'].reshape((1,-1))

    dataset.inputs = Rx[0:1004020]
    dataset.targets = Tx[0:1004020]


    dataset.inputs = np.expand_dims(dataset.inputs, axis=0)
    #dataset.inputs = np.expand_dims(dataset.inputs, axis=0)
    dataset.targets = np.expand_dims(dataset.targets, axis=0)
    #dataset.targets = np.expand_dims(dataset.targets, axis=0)


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
    
    TR_indexes = range(500000) # indexes for training, validation and test set in Piano-midi.de task
    VL_indexes = range(500000,600000)
    TS_indexes = range(600000,800000)
    return dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes
  