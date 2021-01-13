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

myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

Rx = myData['PAMsymRx']#.reshape((1,-1))
Tx = myData['PAMsymTx']#.reshape((1,-1))

datasetRx = Rx[0:20000]#1004020] #set to 1004040 for full dataset
datasetTx = Tx[0:20000]#1004020]


plt.plot(datasetRx,'r-o', markersize = 4, label = "Rx")
plt.plot(datasetTx,'b-o', markersize = 4, label = "Tx")
plt.xlabel("Number of bit")
plt.ylabel("Bit Value")
plt.xlim(50,100)
plt.ylim(-1.5, 1.5)
plt.legend(loc='upper right')
plt.show()



print(mean_squared_error(Tx, Rx))

counter = 0
for x in range(1004020):
    if Rx[x] != Tx[x]:
        counter = counter+1

print('Bit Error Rate', counter/1004020)

#ber = datasetRx/datasetTx

# scalar = MinMaxScaler(feature_range=(-0.5,0.5))

# datasetRx = scalar.fit_transform(datasetRx)
# datasetTx = scalar.fit_transform(datasetTx)

(X_train, X_test, y_train, y_test) = train_test_split(
    datasetRx, datasetTx, test_size = 0.15, random_state=42
) #75% for training 25% for testing



class_names = np.array(["Plus 1","Minus 1"])

batch_size = 64
epochs = 5
model = Sequential()

model.add(Dense(1, activation='softmax')) #first layer
model.add(Dense(128, activation='softmax')) #hidden layers
model.add(Dense(128, activation='softmax'))
model.add(Dense(128, activation='softmax'))
model.add(Dense(2, activation='softmax')) #output layer

opt = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD") #gradient decent

model.compile(loss='mse', optimizer='SGD', metrics=['accuracy']) #compile the model

checkpoint_path = "MLP\model_checkpoints\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#create checkpoint callback
cp_callback =tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)
 #training
history = model.fit(x = X_train, y = y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks = [cp_callback])

test_loss, test_accuracy = model.evaluate(X_test,y_test, verbose=0)

y_pred = model.predict(X_test, batch_size=64, verbose=0) #use test set to find accuracy
#print(y_pred)


y_pred_bool = np.argmax(y_pred, axis=1)
print('\nTest accuracy:', test_accuracy)
print(classification_report(y_test, y_pred_bool, target_names=class_names))


label_numbers = [0,1]
print("Confusion Matrix")
print(print(confusion_matrix(y_test, y_pred_bool, labels = label_numbers)))

def saveResults(predict,input_values, perfect):
    newArr = np.column_stack((predict, input_values, perfect))
    predict = pd.DataFrame(newArr)

    output_filepath = 'MLP\Results\_results_keras.xlsx'
    predict.to_excel(output_filepath, index = False)
  

plt.plot(y_pred, label = "output")
plt.plot(y_test, label = "ideal")

plt.xlim(0,100)
plt.legend(loc='upper right')
plt.show()

print('Test loss:', test_loss)
print('Test Accuracy:', test_accuracy)
saveResults(y_pred, X_test, y_test)




#scalar_end = MinMaxScaler(feature_range=(0,1))
#rescaledTx = scalar_end.fit_transform(y_test)
'''
#mse = (np.square(y_pred - rescaledTx)).mean(axis=None)
#rescaledTx = rescaledTx.flatten()
y_pred = y_pred.flatten()

#print(best)
y_pred = np.round(y_pred) #rounding
#print("Mean Squared Error:", mse)


datasetRx = np.round(datasetRx)


bitError = 0
bitError_noML = 0

datasetRx = datasetRx.flatten()
datasetTx = datasetTx.flatten()


print("Computing bit error Rate")
for x in range(len(y_pred)):
    if datasetRx[x] == datasetTx[x]:
        bitError_noML = bitError_noML +1

    if y_pred[x] == y_test.flatten()[x]:
        bitError = bitError+1


BitErrorRate = bitError/len(y_pred)

BitErrorRate_noML = bitError_noML/len(y_pred)


print("BER ML: ",BitErrorRate)
print("BER no-ML:", BitErrorRate_noML)
'''

