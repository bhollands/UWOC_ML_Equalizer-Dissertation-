import numpy as np
import sys
#sys.path.insert(0, "/Data") #import read data methods
from data_reader import *
import pandas as pd
import openpyxl
from pso import *

#*** ACTIVATION FUNCTIONS***
def activationSigmoid(self, s):
    # activation function
    # simple activationSigmoid curve as in the book
    return 1/(1+np.exp(-s))

def activationHyperbolicTangent(self, s):
    return np.tanh(s)

def activationCosine(self, s):
    return np.cos(s)

def activationGaussian(self, s):
    return np.exp(-((s**2)/2))
##################################################


class FeedForwardNeuralNetwork : #activation function can be selected from outside
    def __init__(self, inLayerSize, outLayerSize, hiddenLayersSize, numOfHiddenLayers, activationFunc, fileSelect):
        self.inputLayer             = inLayerSize #number of neurons in the first layer
        self.outputLayer            = outLayerSize #number of layers in 
        self.numOfHiddenLayers      = numOfHiddenLayers
        self.hiddenLayersSize       = hiddenLayersSize
        self.activationFunc         = activationFunc #the choice of activation function
        self.totalNumberOfLayers    = numOfHiddenLayers + 2 #2 is the input layer and output layer
        self.file_select            = fileSelect
        
        '''
        returns the results of the the chosen activation function
        '''
    def activationFunction(self, value):
        self.activationFunc = self.activationFunc.lower()#make the selection lower case
        if  (self.activationFunc == "sigmoid"): #check if selected is sigmoid
            return activationSigmoid(self, value) #return the value after the 
        elif(self.activationFunc == "hypertangent"): #hyperbolic tangent
            return activationHyperbolicTangent(self,value)
        elif (self.activationFunc == "cosine"): 
            return activationCosine(self,value)
        elif (self.activationFunc == "gaussian"):
            return activationGaussian(self,value)
        else:
            print("Please Enter a Valid activation function:") # if an invalid or misspelling
            print("sigmoid, hypertangent, cosine, gaussian")

    #going ot need to also dynamiclly generate the layers #lotsofdiconaries
    #diconary to look like  = [layer][weight1, weight2, weight3, ..., weightn]
    #chose diconary and not 2d numpy array as there are different sized strucures
    weights = {} #diconary to dynamiclly store the weights, each key'd element is a layer. 
                 #Each array in the array is a neuron and each number is an input weight to thay neuron
    def generate_weights(self):
        print("Generating Intial weights...")
        first = np.random.randn(self.inputLayer,self.hiddenLayersSize) #randn(input, first hidden layer)
        self.weights[0] = first
        for i in range(self.numOfHiddenLayers): # goes through all the hidden layers and puts them all into a diconary
            key = i+1 #starts at 1 not 0 as its already been added
            if(i+1 == self.numOfHiddenLayers): #the the loop has reached the end
                break # break the loop
            else: #else generate the weights and add to diconary
                weightArray = np.random.randn(self.hiddenLayersSize, self.hiddenLayersSize) # creates matrix each set of weights
                self.weights[key] = weightArray
                #print("weight array: ",weightArray)
        last = np.random.randn(self.hiddenLayersSize, self.outputLayer) # the last layer with the hidden layer
        key = self.numOfHiddenLayers
        self.weights[key] = last

    # diconary look like = [layer][bias1, bias2, bias3 ... bias n] biases in an array
    #store the randomly generated bias's in another diconary
    #iconary chosen as the first and last arrays may be of different sizes
    bias_dict = {} #diconary that dynamiclly stores the biases
                    #each key'd element is a layer in the form of an array, each element is a neuron bias
    def generate_bias(self):
        print("Generating Inital bias...")
        for i in range(self.numOfHiddenLayers): #for the number of number of neurons create an array
            key = i #starts from 0
            bias_array = np.random.randn(self.hiddenLayersSize) #creates random biases
            self.bias_dict[key] = bias_array#add to the diconary
        output_layer_bias = np.random.randn(self.outputLayer) #make final layer
        final_key = self.numOfHiddenLayers #find the key
        self.bias_dict[final_key] = output_layer_bias #put into diconary

    '''
    updates the weights and bias of the network from the position of a particle
    '''
    def updateWeightsAndBias(self, p):
        last_pos = 0
        if isinstance(p, Particle): #is p a particle
            p = p.position # the variable p is the position of that particle

        last_pos = self.inputLayer*self.hiddenLayersSize
        self.weights[0] = p[0:last_pos].reshape((self.inputLayer, self.hiddenLayersSize))#get the position data for the length of the number of weights between first and hidden layer
        
        #get bias for first hidden layer
        curr_pos = last_pos #last position is the new starting point
        last_pos = last_pos+self.hiddenLayersSize #last position + the size of the hidden layer
        self.bias_dict[0] = p[curr_pos:last_pos].reshape((self.hiddenLayersSize, ))
        
        for i in range(self.numOfHiddenLayers - 1): #repeat for all the hidden layers
            curr_pos = last_pos
            last_pos = last_pos+self.hiddenLayersSize**2
            self.weights[i+1] = p[curr_pos:last_pos].reshape((self.hiddenLayersSize, self.hiddenLayersSize))

            curr_pos = last_pos
            last_pos = last_pos+self.hiddenLayersSize
            self.bias_dict[i+1] = p[curr_pos:last_pos].reshape((self.hiddenLayersSize,))

        #final weight and bias 
        final_key = self.numOfHiddenLayers
        curr_pos = last_pos
        last_pos = last_pos+(self.hiddenLayersSize*self.outputLayer)
        self.weights[final_key] = p[curr_pos:last_pos].reshape((self.hiddenLayersSize,self.outputLayer))
        curr_pos = last_pos
        last_pos = last_pos+self.outputLayer
        self.bias_dict[final_key] = p[curr_pos:last_pos].reshape((self.outputLayer))

    '''
    returns the mean squared error of a network output
    '''
    def mse(self,y,result):
        loss = np.mean(np.square(y - result)) #mse eqn
        return loss

    '''
    performs the network feed forward
    '''
    def feedForward(self, X,y, particle): #does the feed forward propigation through our network
        #self.weights.clear
        #self.bias_dict.clear
        self.updateWeightsAndBias(particle)#update the weights a bias
        ffwrd3 = None
        ffwrd4 = None
        ffwrd = np.dot(X, self.weights[0])+ self.bias_dict[0] #first layer with the input
        ffwrd2 = self.activationFunction(ffwrd) #put through th activation function
        #print(ffwrd2)
        for i in range(len(self.weights)-1): #loops the number of weight asignments -1 (first already assigned)
            ffwrd3 = np.dot(ffwrd2, self.weights[i+1]) + self.bias_dict[i+1] #do eqn for each layer
            ffwrd4 = self.activationFunction(ffwrd3) #activation funcation
            tmp = ffwrd4 #reassign a3
            ffwrd3 = tmp

        out = ffwrd4
        mse = self.mse(y,out) #get the loss of the output
        return mse

    '''
    initalises the network with the first set of weights and bais in the diconaryies
    '''
    def initNetwork(self):
        self.generate_weights() #generate the inital random weights on createion
        self.generate_bias() #generate theinital bias on creation

    '''
    Finds how many dimensions the search space for the particels is
    '''
    def calcNumOfDims(self):
        first = (self.inputLayer*self.hiddenLayersSize) + self.hiddenLayersSize #dimensions of the first layers of weights and bias
        hidLayers = self.numOfHiddenLayers*((self.hiddenLayersSize**2)+self.hiddenLayersSize) #dimension of the hidden layers
        last = (self.hiddenLayersSize*self.outputLayer)+self.outputLayer #dimension of the last layers
        numOfDim = first + hidLayers + last #add them all together
        return numOfDim
    
    '''
    takes a particle position and input and outputs the network result 
    '''
    def predict(self, X, p):
        #is essentially feedword but does not output the loss
        self.updateWeightsAndBias(p)#update the weights a bias to the best location
        ffwrd3 = None
        ffwrd4 = None

        ffwrd = np.dot(X, self.weights[0])+ self.bias_dict[0]
        ffwrd2 = self.activationFunction(ffwrd)
        #print(ffwrd2)
        for i in range(len(self.weights)-1): #loops the number of weight asignments -1 (first already assigned)
            ffwrd3 = np.dot(ffwrd2, self.weights[i+1]) + self.bias_dict[i+1]
            ffwrd4 = self.activationFunction(ffwrd3)
            tmp = ffwrd4 #reassign a3
            ffwrd3 = tmp
        
        out = ffwrd3
        return out

    '''
    saves and exports the best particle position of the network (weights and bias)
    effectivly saving the optimised network values to be used later
    '''
    def saveBestPos(self, position):
        pos = pd.DataFrame(position) #turn the array into a dataframe
        filePath = 'Best_Particle_position\Best_Particle_Position_'+self.activationFunc+'_'+self.file_select +'.csv' #define the file path
        pos.to_csv(filePath, index = False)#export
        return 0

    '''
    Starts the trainig process, creating the swarm and searchig for the optimal location
    '''
    def train (self, X, y, numParticels,posRange,velRange,iWeightRange,c, epochs, print_step): #modify training class to do it internally
        numOfDim = self.calcNumOfDims() #find the number of dimensions
        swarm  = Swarm(numParticels, numOfDim, posRange, velRange, iWeightRange, c) #make the particle swarm
        fitness_data = swarm.search(self.feedForward, X, y, print_step, epochs, self.file_select) #search for the solution
        best_pos = swarm.get_best_solution() #get the best solution
        self.saveBestPos(best_pos) #save best position
        predication = self.predict(X, best_pos) #get the output
        return (predication,fitness_data) #return the fitness and output



