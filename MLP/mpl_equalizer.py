import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split #to make a % of the dataset for quicker testing
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#need to make train set and test set

in_file = os.path.dirname(os.path.abspath(__file__)) #fix later
Rx_datapath = os.path.join(in_file, 'PAMsymRx.csv')
Tx_datapath = os.path.join(in_file, 'PAMsymTx.csv')


PAMsymTx = pd.read_csv(Tx_datapath)# read in csv data 
PAMsymRx = pd.read_csv(Rx_datapath)

PAMsymTx_Array = PAMsymTx.values.flatten() #turn value into usable numpy array
PAMsymRx_Array = PAMsymRx.values.flatten()

PAMsymTx_Zero_Array = np.where(PAMsymTx_Array==-1,0,PAMsymTx_Array)


(X_train, X_test, y_train, y_test) = train_test_split(
    PAMsymRx_Array, PAMsymTx_Zero_Array, test_size = 0.10, random_state=42
) #75% for training 25% for testing

# np.where(y_train==1,0,y_train)
# np.where(y_test==1,0,y_test)
# print(y_train)
# print(y_test)

batch_size = 64
epochs = 5
model = Sequential()

model.add(Dense(1, activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(16, activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(16, activation='tanh'))
model.add(Dense(16, activation='tanh'))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='tanh'))

opt = tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.0, nesterov=False, name="SGD") #gradient decent

model.compile(loss='mse', optimizer='SGD', metrics=['accuracy'])
#model.summary()

checkpoint_path = "MLP\model_checkpoints\cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

#create checkpoint callback
cp_callback =tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    verbose=1
)



history = model.fit(X_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, y_test),
                    callbacks = [cp_callback])

test_loss, test_accuracy = model.evaluate(X_test,y_test, verbose=0)

y_pred = model.predict(X_test, batch_size=64, verbose=0)
#print(y_pred)


def saveResults(predict,input_values, perfect):
    newArr = np.column_stack((predict, input_values, perfect))
    predict = pd.DataFrame(newArr)

    output_filepath = 'MLP\Results\_results_keras.xlsx'
    #fitness_filepath = 'NN_output data\_fitness_results_'+file +'_'+activeFunc+'.xlsx'
    predict.to_excel(output_filepath, index = False)
    #fitness.to_excel(fitness_filepath, index = False)



saveResults(y_pred, X_test, y_test)

print('Test loss:', test_loss)
print('Test Accuracy:', test_accuracy)


