'''
File for training and testing different Multi-Layer Perceptron networks for Equalisation

On all data set tried and amaounts avg loss = 0.68
'''
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scipy.io import loadmat
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.preprocessing.sequence import TimeseriesGenerator

scaler = MinMaxScaler(feature_range=(0,1))
myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')  
#myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat') 
#myData = loadmat('UWOC27m_PAM16_250Mb_APDGain100_P13dBm.mat')
#myData = loadmat('UWOC27m_PAM16_125Mb_APDGain100_P13dBm.mat')     
#myData = loadmat('UWOC27m_PAM4_125Mb_APDGain80_P7dBm.mat')
#myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')
#myData = loadmat('UWOC27m_PAM8_94Mb_APDGain100_P10dBm.mat') 
#myData = loadmat('UWOC27m_PAM8_188Mb_APDGain100_P10dBm.mat')
#myData = loadmat('UWOC27m_PAM4_125Mb_APDGain80_P7dBm.mat') 



Rx = myData['PAMsymRx']#.reshape((1,-1))
Tx = myData['PAMsymTx']#.reshape((1,-1))
#print(Rx)

#Tx = np.genfromtxt('PAM_OS4_3_6dbm.csv', delimiter=',', usecols=1)
#Rx = np.genfromtxt('PAM_OS4_3_6dbm.csv', delimiter=',', usecols=0)
# print(Rx)

# plt.plot(Tx.flatten(), 'b-o', markersize = 4)#, label="Raw Tx")
# plt.legend(loc='upper right')
# plt.xlabel("Number of bit")
# plt.ylabel("Bit status")
# plt.xlim([100,150])
# #plt.grid()
# plt.show()




rescale = True

if rescale:
    Tx = Tx.reshape(-1,1)
    Rx = Rx.reshape(-1,1)
    #print(Tx)

    Rx = scaler.fit_transform(Rx)
    Tx = scaler.fit_transform(Tx)


datasetRx = Rx[0:240000].flatten()#1004020] #set to 1004040 for full dataset
datasetTx = Tx[0:240000].flatten()#1004020]
rx = datasetRx
tx = datasetTx

###print(datasetRx)

input_size =8

'''
for i in range(len(gen)):
	x, y = gen[i]
	print('%s => %s' % (x, y))
'''
X_train = rx[0:100000]
y_train = tx[0:100000]

X_test = rx[140000:240000]
y_test = tx[140000:240000]



# (X_train, X_test, y_train, y_test) = train_test_split(
#     rx, tx, test_size = 0.5, random_state=42, shuffle = False
# ) #75% for training 25% for testing

print(X_train)

gen1 = TimeseriesGenerator(X_train, y_train,length = input_size, batch_size=32)
gen2 = TimeseriesGenerator(X_test, y_test,length = input_size, batch_size=32)

batch_size = 32
epochs = 15
model = Sequential()

layer_size = 40
model.add(Dense(layer_size, activation='selu', input_dim = input_size)) #first layer
model.add(Dense(layer_size, activation='selu')) #2
model.add(Dense(layer_size, activation='selu'))#3
model.add(Dense(layer_size, activation='selu'))#4
# model.add(Dense(layer_size, activation='selu'))#5
# model.add(Dense(layer_size, activation='selu'))#6
# model.add(Dense(layer_size, activation='selu'))#7
# model.add(Dense(layer_size, activation='selu'))#8
#model.add(Dense(layer_size, activation='selu'))#9
#model.add(Dense(layer_size, activation='selu'))#10

model.add(Dense(1)) #output layer


#opt = tf.keras.optimizers.SGD(
#    learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD") #gradient decent

model.compile(loss='mse', optimizer='adam', metrics=['accuracy']) #compile the model

checkpoint_path = "MLP\model_checkpoints\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#create checkpoint callback
cp_callback =tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)
 #training
# history = model.fit(x = X_train, y = y_train,
#                     batch_size=batch_size,
#                     epochs=epochs,
#                     verbose=1,
#                     validation_data=(X_test, y_test),
#                     callbacks = [cp_callback])

model.fit(gen1, epochs=epochs, verbose=1)

test_loss, test_accuracy = model.evaluate(gen2, verbose=1)

y_pred = model.predict(gen2) #use test set to find accuracy

def saveResults(predict,input_values, perfect):
    newArr = np.column_stack((predict, input_values, perfect))
    predict = pd.DataFrame(newArr)

    output_filepath = 'MLP\Results\_results_keras.xlsx'
    predict.to_excel(output_filepath, index = False)
  
def calcBer(Rx, Tx):
    error = 0
    for x in range(Rx.size):
        if Rx[x] != Tx[x]:
            error  = error +1
    
    return error/Rx.size

def classify2PAM(array):
    for i in range(array.size):
        if array[i] >= 0.5:
            array[i] = 1
        else:
            array[i] = 0
    return array

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
                    #print("Error:", a[y], ' != ', b[y] )
                    localError = localError + 1
            if localError >= 2: #if there are 2 or more error
                bitError = bitError +1 #the bit has failed
            localError = 0
            
    ber = bitError/size
    return ber 

def back_to_original(array):
    for i in range(array.size):
        if array[i] == 1:
            array[i] = 8191
        elif array[i] == 0.6666463191307532:
            array[i] = 2730
        elif array[i] == 0.33335368086924677:
            array[i] = -2730
        elif array[i] == 0:
            array[i] = -8191
    return array


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scikitplot as skplt
import pandas as pd
import seaborn as sns
#y_pred = classify4PAM(y_pred.flatten())
#print(y_pred)
#y_pred = back_to_original(y_pred)
# print(y_pred)

#y_test = classify4PAM(y_test.flatten())
#ber = pam4Ber(y_pred.flatten(), y_test.flatten())
print(y_pred.shape)
print(y_test.shape)

from scipy.io import savemat
#myData = loadmat('UWOC27m_PAM4_125Mb_APDGain80_P7dBm.mat')
#myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')
orig_Rx = myData['PAMsymRx']#.reshape((1,-1))#Mat']#.reshape((1,-1))
orig_Tx = myData['PAMsymTx']#.reshape((1,-1))#Mat']#.reshape((1,-1))


equalized = y_pred.reshape(-1,1)
Tx = orig_Tx[140000+input_size:240000].reshape(-1,1)


Input =orig_Rx[140000+input_size:240000].reshape(-1,1)

if rescale:
    equalized = scaler.inverse_transform(equalized)

#Tx = scalar2.fit_transform(Tx)
mdic = {"Rx": equalized, "Tx": Tx, "input":Input}
savemat("equalizedOutput.mat", mdic)

#print("Bit Error Rate:", ber)
from sklearn.metrics import mean_absolute_percentage_error

print('Test loss:', test_loss)
print('Test Accuracy:', test_accuracy)


plt.plot(Tx.flatten(), 'r-o', markersize = 4, label="Tx")
plt.plot(equalized.flatten(), 'b-o', markersize = 4, label="Equalized Rx")
plt.plot(Input.flatten(), 'g-o', markersize = 4, label="Rx")

#print("y_test", y_test)
# plt.plot(y_pred.flatten(), label = "y_pred")
# #plt.plot(y_test.flatten(), label = "y_test")
# plt.plot(equalized.flatten(), label = "equalised")
# plt.plot(Tx.flatten(), label = "Tx")

plt.xlabel("Number of Symbol")
plt.ylabel("Symbol status")
plt.xlim(0,50)
plt.ylim([-1.5, 1.5])
plt.grid()
plt.title('2PAM MLP Signal Snapshot')

plt.legend(loc='upper right')
#skplt.metrics.plot_confusion_matrix(y_test, y_pred, normalize = False, cmap=plt.cm.Blues)
plt.show()

# #saveResults(y_pred, X_test, y_test)


