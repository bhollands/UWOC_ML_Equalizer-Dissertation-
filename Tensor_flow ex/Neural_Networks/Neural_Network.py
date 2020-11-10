#https://colab.research.google.com/drive/1m2cg3D1x3j5vrFc-Cu0gMvc48gWyCOuG#forceEdit=true&sandboxMode=true

#Tensor flow and keras
import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt 

#Importing the dataset 
fashion_mnist = keras.datasets.fashion_mnist #load the dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#look at the datasets
train_images.shape
train_images[0,23,23] #look at 1 pixel

'''
Our pixel values are between 0 and 255, 0 being black and 255 being white. This means we have a grayscale image as there are no color channels.
'''
print(train_labels[:10]) #have a look at the first 10 training labels

'''
Our labels are integers ranging from 0 - 9. Each integer represents a specific article of clothing. We'll create an array of label names to indicate which is which.
'''
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

'''
Fianlly let's look at what some of these images look like!
'''
#plt.figure()
#plt.imshow(train_images[1])
#plt.colorbar()
#plt.grid(False)
#plt.show()

'''
Data pre-processing
changeing 0-255 to be between 0 and 1
'''

train_images = train_images/255
test_images = test_images/255

#Building the model
'''
Now it's time to build the model! We are going to use a keras sequential model with three different layers. 
This model represents a feed-forward neural network (one that passes values from left to right). 
We'll break down each layer and its architecture below.
'''
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),       #input layer (1)
    keras.layers.Dense(128, activation='relu'),     #hidden layer (2)
    keras.layers.Dense(10, activation='softmax')    #output layer (3)
])
'''
Layer 1: This is our input layer and it will conist of 784 neurons. We use the flatten layer with an input shape of (28,28) to denote that our input should come in in that shape. 
The flatten means that our layer will reshape the shape (28,28) array into a vector of 784 neurons so that each pixel will be associated with one neuron.

Layer 2: This is our first and only hidden layer. The dense denotes that this layer will be fully connected and each neuron from the previous layer connects to each neuron of this layer. 
It has 128 neurons and uses the rectify linear unit activation function.

Layer 3: This is our output later and is also a dense layer. It has 10 neurons that we will look at to determine our models output. 
Each neuron represnts the probabillity of a given image being one of the 10 different classes. The activation function softmax is used on this layer to calculate a probabillity distribution for each class. 
This means the value of any neuron in this layer will be between 0 and 1, where 1 represents a high probabillity of the image being that class.
'''

#Compile the model
'''
The last step in building the model is to define the loss function, optimizer and metrics we would like to track. I won't go into detail about why we chose each of these right now.
'''
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the model
'''
Now it's finally time to train the model. Since we've already done all the work on our data this step is as easy as calling a single method.
'''
model.fit(train_images, train_labels, epochs=10)  # we pass the data, labels and epochs and watch the magic!
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=0) 
print('Test accuracy: ', test_acc)

#Make predictions
'''
To make predictions we simply need to pass an array of data in the form we've specified in the input layer to .predict() method
'''
predications = model.predict(test_images)
'''
This method returns to us an array of predictions for each image we passed it. Let's have a look at the predictions for image 1.
'''
predications[0]
'''
If we wan't to get the value with the highest score we can use a useful function from numpy called argmax(). This simply returns the index of the maximium value from a numpy array. 
'''
np.argmax(predications[0])
'''
And we can check if this is correct by looking at the value of the cooresponding test label.
'''
test_labels[0]

#Verifying Predictions
'''
I've written a small function here to help us verify predictions with some simple visuals.
'''

COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    prediction = model.predict(np.array([image]))
    predicted_class = class_names[np.argmax(prediction)]
    show_image(image, class_names[correct_label], predicted_class)

def show_image(img, label, guess):
  print("Excpected: " + label)
  print("Guess: " + guess)
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)

  plt.colorbar()
  plt.grid(False)
  plt.show()

def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)

            
    