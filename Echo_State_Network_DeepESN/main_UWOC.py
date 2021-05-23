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
from scipy.io import loadmat, savemat
from utils import MSE, config_UWOC, load_UWOC, select_indexes, MAPE
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import time
import itertools

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
    path = ''
    dataset, Nu, error_function, optimization_problem, TR_indexes, VL_indexes, TS_indexes, scaler, data_set_name = load_UWOC(path, MSE)
 
    # load configuration for UWOC task
    config = config_UWOC(list(TR_indexes) + list(VL_indexes))
    # Be careful with memory usage dummy
    recc_units = 10 # number of recurrent units
    recc_layer = 17 # number of recurrent layers
     
    reg = 0
    transient = 100
    deepESN = DeepESN(Nu, recc_units, recc_layer, config)

    states = deepESN.computeState(dataset.inputs, deepESN.IPconf.DeepIP)

    train_states = select_indexes(states, list(TR_indexes) + list(VL_indexes), transient)
    train_targets = select_indexes(dataset.targets, list(TR_indexes) + list(VL_indexes), transient)
    
    test_targets = select_indexes(dataset.targets, TS_indexes)
    start_time_tr = time.time()
    deepESN.trainReadout(train_states, train_targets, reg) #train the network
    end_time_tr = time.time() - start_time_tr
    print('Training Time:', end_time_tr)
    
    train_outputs = deepESN.computeOutput(train_states)


    train_error = error_function(X = train_outputs,Y = train_targets)
    print('Training MSE: ', np.mean(train_error), '\n')

    start_time = time.time()
    test_states = select_indexes(states, TS_indexes)
    test_outputs = deepESN.computeOutput(test_states)
    end_time = time.time() - start_time

    
    test_error = error_function(X = test_outputs, Y = test_targets)
    print('Test MSE: ', np.mean(test_error), '\n')
    print("Time to put unseen data through : %s seconds" % (end_time))
 

    test_targets = np.array(test_targets)
    test_targets = test_targets.flatten()

    test_states = np.array(test_states)
    test_states = test_states.flatten()

    from scipy.io import savemat, loadmat
    #test_outputs
    #test_targets
    myData = loadmat(data_set_name) 
    orig_Rx = myData['PAMsymRx']#.reshape((1,-1))#Mat']#.reshape((1,-1))
    orig_Tx = myData['PAMsymTx']#.reshape((1,-1))#Mat']#.reshape((1,-1))
   
    equalized = test_outputs.reshape(-1,1)
    tr = 20000
    Tx = orig_Tx[tr:240000].reshape(-1,1)
    Input = orig_Rx[tr:240000].reshape(-1,1)

    equalized = scaler.inverse_transform(equalized)

    mdic = {"Rx": equalized, "Tx": Tx ,"input":Input}
    savemat("equalizedOutput.mat", mdic)

    from sklearn.metrics import classification_report
    from sklearn.metrics import confusion_matrix
    import scikitplot as skplt
    import pandas as pd
    import seaborn as sns
    
    plt.plot(Tx.flatten(), 'r-o', markersize = 4, label="Tx")
    plt.plot(equalized.flatten(), 'b-o', markersize = 4, label="Equalized Rx")
    plt.plot(Input.flatten(), 'g-o', markersize = 4, label="Rx")
    plt.legend(loc='upper right')
    plt.xlabel("Number of Symbol")
    plt.ylabel("Symbol Status")
    plt.xlim([0,50])
    plt.grid()
    #plt.show()

    #print("Layers:", Nl)
    #back_to_original(test_outputs_classified)

    # test_targets_orig = back_to_original(test_targets)
    # class_names = ["8191", "2730","-2730","-8191"]
    # label_numbers = [8191,2730,-2730,-8191]
    #class_names = ["0", "1"]
    #label_numbers = [0,1]

    #print(classification_report)


    #report = classification_report(test_targets, test_outputs_classified, digits = 5, target_names=class_names, output_dict = False)

    #print(report)
    #sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, linewidths=.5)
    #plot_classification_report(report)


    # print("Confusion Matrix")
    # print(print(confusion_matrix(test_targets, test_outputs_classified, labels = label_numbers)))

    # skplt.metrics.plot_confusion_matrix(test_targets, test_outputs_classified, normalize = False, cmap=plt.cm.Blues)
    
    

if __name__ == "__main__":
    main()
    
