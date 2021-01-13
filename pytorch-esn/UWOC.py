import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
from sklearn.preprocessing import MinMaxScaler
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing

myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

Rx = myData['PAMsymRx']#.reshape((1,-1))
Tx = myData['PAMsymTx']#.reshape((1,-1))

datasetRx = Rx[0:5000]#1004020] #set to 1004040 for full dataset
datasetTx = Tx[0:5000]#1004020]

#scalar = MinMaxScaler(feature_range=(0,1))

#datasetRx = scalar.fit_transform(datasetRx)
#datasetTx = scalar.fit_transform(datasetTx)

datasetRx = datasetRx.reshape(-1, 1, 1)
datasetTx = datasetTx.reshape(-1, 1, 1)

X_data = torch.from_numpy(datasetRx)
Y_data = torch.from_numpy(datasetTx)

(trainX, testX, trainY, testY) = train_test_split(
    X_data, Y_data, test_size = 0.2, random_state=42
) #75% for training 25% for testing


washout = [500]
input_size = output_size = 1
hidden_size = 1000#

loss_fcn = torch.nn.MSELoss()


if __name__ == "__main__":
    start = time.time()

    #Training
    trainY_flat = utils.prepare_target(trainY.clone(), [trainX.size(0)], washout)


    model = ESN(input_size, hidden_size, output_size)
    model(trainX, washout, None, trainY_flat)
    model.fit()

    output, hidden = model(trainX, washout)

    print("Training error:", loss_fcn(output, trainY[washout[0]:]).item())

    # Test
    output, hidden = model(testX, [0], hidden)
    print("Test error:", loss_fcn(output, testY).item())

    
    np_out      = output.detach().numpy().flatten()
    np_in       = testX.detach().numpy().flatten()
    np_ideal    = testY.detach().numpy().flatten()


    unmod_MSE = (np.square(np_ideal - np_in )).mean(axis=None)
    print("Untampered MSE: ", unmod_MSE)
    print("Ended in", time.time() - start, "seconds.")
    
    
    plt.plot(np_out,'g-o',markersize=4, label = "output")#linestyle = '-')
    plt.plot(np_ideal,'b-o',markersize=4, label = "ideal")#linestyle = '-')
    plt.plot(np_in, 'r-o',markersize=4, label = "input", linestyle = '--', linewidth = '1')
    plt.xlabel('Number of bits')
    plt.ylabel('Bit status')
    plt.grid(True)
    plt.xlim(0,50)
    #plt.ylim(-1,2)
    plt.legend(loc='upper right')
    plt.show()
