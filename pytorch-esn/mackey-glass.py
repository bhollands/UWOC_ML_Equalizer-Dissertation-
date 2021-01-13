import torch.nn
import numpy as np
from torchesn.nn import ESN
from torchesn import utils
import time
import matplotlib.pyplot as plt

#device = torch.device('cuda')
dtype = torch.double
torch.set_default_dtype(dtype)

if dtype == torch.double:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float64)
elif dtype == torch.float:
    data = np.loadtxt('datasets/mg17.csv', delimiter=',', dtype=np.float32)

# plt.plot(data)
# plt.xlim(0,100)
# plt.show()

X_data = np.expand_dims(data[:, [0]], axis=1)
Y_data = np.expand_dims(data[:, [1]], axis=1)
print(X_data)

X_data = torch.from_numpy(X_data)#.to(device)
Y_data = torch.from_numpy(Y_data)#.to(device)

#train test split
trainX = X_data[:5000] #trainX
trainY = Y_data[:5000] #trainy
testX = X_data[5000:] #testX
testY = Y_data[5000:] #testy

washout = [500]
input_size = output_size = 1
hidden_size = 500
loss_fcn = torch.nn.MSELoss()

if __name__ == "__main__":
    start = time.time()

    # Training
    trainY_flat = utils.prepare_target(trainY.clone(), [trainX.size(0)], washout)

    model = ESN(input_size, hidden_size, output_size)
    #model.to(device)

    model(trainX, washout, None, trainY_flat)
    model.fit()
    output, hidden = model(trainX, washout)
    print("Training error:", loss_fcn(output, trainY[washout[0]:]).item())

    # Test
    output, hidden = model(testX, [0], hidden)
    print("Test error:", loss_fcn(output, testY).item())
    print("Ended in", time.time() - start, "seconds.")
