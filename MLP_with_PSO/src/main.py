from data_reader import *
from network import *
import pandas

'''
saves the results of a run
'''
def saveResults(predict, actual, activeFunc, file, fitness):
    newArr = np.column_stack((actual, predict))
    predict = pd.DataFrame(newArr)
    fitness = pd.DataFrame(fitness)

    output_filepath = 'NN_output data\_results_'+file +'_'+activeFunc+'.xlsx'
    fitness_filepath = 'NN_output data\_fitness_results_'+file +'_'+activeFunc+'.xlsx'
    predict.to_excel(output_filepath, index = False)
    fitness.to_excel(fitness_filepath, index = False)

'''
configures some aspects of the network
'''
def configure(file_select):
    data = data_reader() #Create new data reader
    single_input = data.select(file_select) #select the file and it returns if it is single input
    X = data.input_array() #input input array
    y = data.expected_output_array() #get function output

    if single_input: #if network is single input one input we only need single input neuron
        input_layer = 1
    else: #else it is 2 
        input_layer = 2
    return(X,y,input_layer) #return relavent data

'''
Runs all possible data and activation function combinations
'''
def recordAllPossible(hiddenLayersSize, numOfHidden, save, epochs,numParticels,posRange, velRange, coef, iWeightRange):
    #do not call single run as we want to save all results in a different way
    for j in range(6):
        if j == 0: file_select = "cubic"
        if j == 1: file_select = "linear"
        if j == 2: file_select = "sine"
        if j == 3: file_select = "tanh"
        if j == 4: file_select = "complex"
        if j == 5: file_select = "xor"
        #data imports
        config = configure(file_select)

        for i in range(4): ##need to do as function chnages with run and need a fresh network
            if i == 0:
                function = "sigmoid"
                nn = FeedForwardNeuralNetwork(inLayerSize=config[2] ,outLayerSize=1,hiddenLayersSize=hiddenLayersSize,
                                                numOfHiddenLayers=numOfHidden, activationFunc=function, fileSelect = file_select)
                nn.initNetwork() #create the network
                sigmoid = nn.train(X = config[0], y = config[1],numParticels=numParticels, posRange=posRange, 
                                velRange=velRange, iWeightRange=iWeightRange,
                                c=coef, epochs= epochs, print_step=epochs/10) #train on x and y for 1 iteration
            if i == 1:
                function = "hypertangent"
                nn = FeedForwardNeuralNetwork(inLayerSize=config[2] ,outLayerSize=1,hiddenLayersSize=hiddenLayersSize,
                                                numOfHiddenLayers=numOfHidden, activationFunc=function, fileSelect = file_select)
                nn.initNetwork() #create the network
                hypertangent = nn.train(X = config[0], y = config[1],numParticels=numParticels, posRange=posRange, 
                                velRange=velRange, iWeightRange=iWeightRange,
                                c=coef, epochs= epochs, print_step=epochs/10) #train on x and y for 1 iteration
            if i == 2:
                function = "cosine"
                nn = FeedForwardNeuralNetwork(inLayerSize=config[2] ,outLayerSize=1,hiddenLayersSize=hiddenLayersSize,
                                                numOfHiddenLayers=numOfHidden, activationFunc=function, fileSelect = file_select)
                nn.initNetwork() #create the network
                cosine = nn.train(X = config[0], y = config[1],numParticels=numParticels, posRange=posRange, 
                                velRange=velRange, iWeightRange=iWeightRange,
                                c=coef, epochs= epochs, print_step=epochs/10) #train on x and y for 1 iteration
            if i == 3:
                function = "gaussian"
                nn = FeedForwardNeuralNetwork(inLayerSize=config[2] ,outLayerSize=1,hiddenLayersSize=hiddenLayersSize,
                                                numOfHiddenLayers=numOfHidden, activationFunc=function, fileSelect = file_select)
                nn.initNetwork() #create the network
                gaussian = nn.train(X = config[0], y = config[1],numParticels=numParticels, posRange=posRange, 
                                velRange=velRange, iWeightRange=iWeightRange,
                                c=coef, epochs= epochs, print_step=epochs/10) #train on x and y for 1 iteration

        output_result = np.column_stack((config[1], sigmoid[0], hypertangent[0], cosine[0], gaussian[0],)) #get all the results in a single array
        predict = pd.DataFrame(output_result) #turn into dataframe
        output_filepath = 'NN_output data\_results_'+file_select +'.xlsx' #define filepath
        predict.to_excel(output_filepath, index = False) #save as excel file

'''
runs a single specificed configuration
'''
def singleRun(file, activationFunc, hiddenLayersSize, numOfHidden, save, epochs,numParticels,posRange, velRange, coef, iWeightRange):
    config = configure(file) #configure the network

    nn = FeedForwardNeuralNetwork(inLayerSize=config[2] ,outLayerSize=1,hiddenLayersSize=hiddenLayersSize,
                                    numOfHiddenLayers=numOfHidden, activationFunc=activationFunc, fileSelect = file) # creat the network
    nn.initNetwork() #initalise the network
    result = nn.train(X = config[0], y = config[1],numParticels=numParticels, posRange=posRange, 
                    velRange=velRange, iWeightRange=iWeightRange,
                    c=coef, epochs= epochs, print_step=epochs/10) #train on x and y for given number of iterations

    if save == True:
        saveResults(result[0], config[1], activationFunc, file, result[1]) #save the results of the run
        #saveFitness(result[1], file)



def main():
    # singleRun(file="complex", activationFunc="cosine", hiddenLayersSize = 5, 
    #             numOfHidden = 2, save=True, epochs=50, numParticels=200,  
    #             posRange=(-2.0,2.0), velRange=(-2.0, 2.0), 
    #             coef = (0.5,0.3), iWeightRange = (0.9, 0.9))

    recordAllPossible(hiddenLayersSize = 5, numOfHidden = 2, save=True, 
                epochs=50, numParticels=200,  
                posRange=(-2.0,2.0), velRange=(-2.0, 2.0), 
                coef = (0.5,0.3), iWeightRange = (0.9, 0.9))


if __name__ == "__main__":
    main()