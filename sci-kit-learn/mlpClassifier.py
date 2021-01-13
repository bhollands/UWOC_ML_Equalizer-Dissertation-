from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


myData = loadmat('POF60m_PAMExp_2PAM_DR600Mbps(single column).mat')   

Rx = myData['PAMsymRx']#.reshape((1,-1))
Tx = myData['PAMsymTx']#.reshape((1,-1))

datasetRx = Rx[0:20000]#1004020] #set to 1004040 for full dataset
datasetTx = Tx[0:20000]#1004020]

(X_train, X_test, y_train, y_test) = train_test_split(
    datasetRx, datasetTx, test_size = 0.05, random_state=42
) #75% for training 25% for testing

clf = MLPClassifier(hidden_layer_sizes=(40), activation='tanh', solver = 'adam', 
                    learning_rate = 'adaptive', max_iter=300).fit(X_train, y_train)


output = clf.predict(X_test)

Accuracy = clf.score(X_test, y_test)
print("Accuracy = ", Accuracy)



ber = 1.0 - Accuracy
loss = mean_squared_error(y_test, output)


print("Bit Error Rate = ", ber)
print("Loss = ", loss)
plt.plot(output.flatten(),"g-o", markersize = 4, label = "output")
plt.plot(y_test.flatten(),"b-o", markersize = 4, label = "ideal")
plt.xlabel("Number of bit")
plt.ylabel("Bit Value")


plt.xlim(0,50)
plt.legend(loc='upper right')
plt.show()
