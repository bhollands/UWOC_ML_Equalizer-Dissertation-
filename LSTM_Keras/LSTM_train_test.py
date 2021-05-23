
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
from scipy.io import loadmat
import matplotlib.pyplot as plt
import math

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.layers.core import Activation

'''
Pre-processing
'''
myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   
#myData = loadmat('UWOC27m_PAM4_125Mb_APDGain80_P7dBm.mat') 
#myData = loadmat('UWOC27m_PAM8_94Mb_APDGain100_P10dBm.mat')
#myData = loadmat('UWOC27m_PAM8_188Mb_APDGain100_P10dBm.mat') 
#myData = loadmat('UWOC27m_PAM16_250Mb_APDGain100_P13dBm.mat') 

Rx = myData['PAMsymRx']#.reshape((1,-1))
Tx = myData['PAMsymTx']#.reshape((1,-1))

#Tx = np.genfromtxt('PAM_OS4_3_6dbm.csv', delimiter=',', usecols=1)
#Rx = np.genfromtxt('PAM_OS4_3_6dbm.csv', delimiter=',', usecols=0)

def pam4Ber(toCheck, perfect):
    bitError = 0
    localError = 0
    toCheck = toCheck.reshape(-1,4)
    perfect = perfect.reshape(-1,4)
    size = int(perfect.size/4)
    for i in range(size):#go through entir dataset
        a = toCheck[i]
        b = perfect[i]
        if not np.array_equal(a,b): #if they are not equal
            for y in range(a.size):
                if a[y] != b[y]: #see how many are not equal
                    localError = localError + 1
            if localError >= 2: #if there are 2 or more error
                bitError = bitError +1 #the bit has failed
            localError = 0
            
    ber = bitError/size
    return ber 


def classify4PAM(array):
    for i in range(array.size):
        if array[i] >= 0.83329264:
            array[i] = 3
        elif array[i]  >= 0.5 and array[i] < 0.83329264:
            array[i] = 2#0.6666463191307532
        elif array[i]  > 0.16667684 and array[i] < 0.5:
            array[i] = 1#0.33335368086924677
        elif array[i]  <= 0.16667684:
            array[i] = 0

    return array


PAMsymRx_Array = Rx[0:247494].reshape(-1,1) #set to 1004040 for full dataset
PAMsymTx_Array = Tx[0:247494].reshape(-1,1) 

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

rescale = True

if rescale:
    scalar = MinMaxScaler(feature_range=(0,1))
    PAMsymRx_Array = scalar.fit_transform(PAMsymRx_Array)
    PAMsymTx_Array = scalar.fit_transform(PAMsymTx_Array)

train = 80000

X_train = PAMsymRx_Array[0:train]
y_train = PAMsymTx_Array[0:train]
X_test = PAMsymRx_Array[140000:240000]
y_test = PAMsymTx_Array[140000:240000]

X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
y_train = np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
y_test = np.reshape(y_test, (y_test.shape[0], 1, y_test.shape[1]))




PAMsymRx_Array = np.reshape(PAMsymRx_Array, (PAMsymRx_Array.shape[0], 1, PAMsymRx_Array.shape[1]))
PAMsymTx_Array = np.reshape(PAMsymTx_Array, (PAMsymTx_Array.shape[0], 1, PAMsymTx_Array.shape[1]))

def calcBer(Rx, Tx):
    error = 0
    for x in range(Rx.size):
        if Rx[x] != Tx[x]:
            error  = error +1
    
    return error/Rx.size


def accuracy(toCheck, perfect):
    correct = 0
    for i in range(perfect.size):
        if toCheck[i] == perfect[i]:
            correct = correct +1
    ber = correct/perfect.size
    return ber


model = Sequential()
model.add(LSTM(5, input_shape=(1,1)))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=2, batch_size=32, verbose=1)

test_loss, test_accuracy = model.evaluate(X_test,y_test, verbose=0)

# plt.plot(history.history['loss'])

# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# #plt.show()
y_pred = model.predict(X_test) #use test set to find accuracy

#y_pred = y_pred.flatten()
#classified = classify4PAM(y_pred)
#ber = pam4Ber(classified, y_test.flatten())
#ber = calcBer(y_pred, y_test)
#print("Bit Error Rate:", ber)
#accuracy = accuracy(classified, y_test)
#print(accuracy)
print('Test loss:', test_loss)
print('Test Accuracy:', test_accuracy)



# y_pred = back_to_original(y_pred)
# print(y_pred)

from scipy.io import savemat

Rx = y_pred.reshape(-1,1)
Tx = y_test.reshape(-1,1)
#Input = X_test.reshape(-1,1)
if rescale:
    Rx = scalar.inverse_transform(Rx)
    Tx = scalar.inverse_transform(Tx)

Input = myData['PAMsymRx'][140000:240000]

mdic = {"Rx": Rx, "Tx": Tx, "input": Input}
savemat("equalizedOutput.mat", mdic)


plt.plot(Tx.flatten(), 'r-o', markersize = 4, label="Tx")
plt.plot(Rx.flatten(), 'g-o', markersize = 4, label="Equalized Rx")
plt.plot(Input.flatten(), 'b-o', markersize = 4, label="Rx")
plt.legend(loc='upper right')
plt.xlabel("Number of bit")
plt.ylabel("Bit status")
plt.xlim([100,150])
plt.ylim([-1.5, 1.5])
plt.grid()
plt.title('2PAM LSTM Signal Snapshot')
#plt.show()



