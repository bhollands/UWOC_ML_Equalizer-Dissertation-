'''
This is an example of the application of DeepESN model for next-step prediction on Mackey Glass time-series. 

Reference paper for DeepESN model:
C. Gallicchio, A. Micheli, L. Pedrelli, "Deep Reservoir Computing: A Critical Experimental Analysis", 
Neurocomputing, 2017, vol. 268, pp. 87-99

Reference paper for the design of DeepESN model in multivariate time-series prediction tasks:
C. Gallicchio, A. Micheli, L. Pedrelli, "Design of deep echo state networks",
Neural Networks, 2018, vol. 108, pp. 33-47 
    
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

import numpy as np
import pandas as pd
import random
from DeepESN import DeepESN
from scipy.io import loadmat
from utils import MSE, config_MG, load_UWOC, select_indexes 
import matplotlib.pyplot as plt
class Struct(object): pass

def saveOutputResults(predict):
    newArr = np.column_stack((predict))
    predict = pd.DataFrame(newArr)

    output_filepath = 'Echo_State_Network_DeepESN\Results\_results_DeepESN.xlsx'
    #fitness_filepath = 'NN_output data\_fitness_results_'+file +'_'+activeFunc+'.xlsx'
    predict.to_excel(output_filepath, index = False)
    #fitness.to_excel(fitness_filepath, index = False)
# sistemare indici per IP in config_pianomidi, mettere da un'altra parte
# sistema selezione indici con transiente messi all'interno della rete

def plotResults(predict):
    plt.plot(predict, label="results")
    plt.legend(loc='upper right')
    plt.xlim([0,100])
    plt.show()


def main():

    # fix a seed for the reproducibility of results
    np.random.seed(7)

    # dataset path 
    path = 'Echo_State_Network_DeepESN\datasets'
    dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes = load_UWOC(path, MSE)

    # load configuration for pianomidi task
    configs = config_MG(list(TR_indexes) + list(VL_indexes))
    # Be careful with memory usage
    Nr = 25 # number of recurrent units
    Nl = 10 # number of recurrent layers
    reg = 0.0
    transient = 100
    deepESN = DeepESN(Nu, Nr, Nl, configs)
    states = deepESN.computeState(dataset.inputs, deepESN.IPconf.DeepIP)
    train_states = select_indexes(states, list(TR_indexes) + list(VL_indexes), transient)
    train_targets = select_indexes(dataset.targets, list(TR_indexes) + list(VL_indexes), transient)
    
    test_states = select_indexes(states, TS_indexes)
    test_targets = select_indexes(dataset.targets, TS_indexes)
    
    deepESN.trainReadout(train_states, train_targets, reg)
    
    train_outputs = deepESN.computeOutput(train_states)
    train_error = error_function(train_outputs, train_targets)
    print('Training MSE: ', np.mean(train_error), '\n')
    test_outputs = deepESN.computeOutput(test_states)
   
    
    test_error = error_function(test_outputs, test_targets)
    print('Test MSE: ', np.mean(test_error), '\n')
    saveOutputResults(test_outputs)

    test_targets = np.array(test_targets)


    plt.plot(test_targets.flatten(), 'b-o', markersize = 4, label="Ideal ")
    plt.plot(test_outputs.flatten(), 'r-o', markersize = 4, label="Output")
    plt.plot(test_states, 'g-o', markersize = 4, label="Input")
    plt.legend(loc='upper right')
    plt.xlabel("Number of bit")
    plt.ylabel("Bit status")
    plt.xlim([0,50])
    plt.show()

if __name__ == "__main__":
    main()
    