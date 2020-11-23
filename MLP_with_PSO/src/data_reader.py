import pandas as pd
import numpy as np

class data_reader:  
    one_input = False      
    pandas_data = None

    def select(self, file_select): #the user says what data they want to use
        print("File Selected =", file_select)
        correctSpelling = False # spelling is always false as user hasnt entered anyting
        if file_select == "cubic":
            self.one_input = True
            correctSpelling = True
            DataSetPath = "src\Data\OneIn_cubic.csv"
        elif file_select == "linear":
            self.one_input = True
            correctSpelling = True
            DataSetPath = "src\Data\OneIn_linear.csv"
        elif file_select == "sine":
            self.one_input = True
            correctSpelling = True
            DataSetPath = "src\Data\OneIn_sine.csv"
        elif file_select == "tanh":
            self.one_input = True
            correctSpelling = True
            DataSetPath = "src\Data\OneIn_tanh.csv"
        elif file_select == "complex":
            correctSpelling = True
            DataSetPath = "src\Data\TwoIn_complex.csv"
        elif file_select == "xor":
            correctSpelling = True
            DataSetPath = "src\Data\TwoIn_xor.csv"
        # elif file_select == "user_custom_data":
        #     self.one_input = True #if single input
        #     correctSpelling = True
        #     DataSetPath = "src\Data\user_custom_data.csv"
        else:
            correctSpelling = False
            print("Please enter an available dataset")

        self.pandas_data = pd.read_csv(DataSetPath)
        return self.one_input

    '''
    turns pandas dataframe to numpy array
    '''
    def pandas_to_numpy(self, data):
        df1 = pd.DataFrame(data)
        data_numpy = df1.values
        return data_numpy
   
    '''
    makes 1D array
    '''
    def make_single_array(self, data, column, sizeX):
        to_fill = np.zeros(sizeX)
        for i in range(sizeX):
            to_fill[i] = data[i][column]
        return to_fill

    '''
    makes 2D array for input
    '''
    def make_double_in_array(self, data):
        to_fill = np.zeros(shape = (100, 2)) #will always be that shape
        for i in range (100): #will always be 99 with this dataset
            to_fill[i][0] = data[i][0]
            to_fill[i][1] = data[i][1]
        return to_fill
    
    '''
    returns the input array
    '''
    def input_array(self):
        data_array = self.pandas_to_numpy(self.pandas_data) #turns the read input dataframe ot numpy array
        size = int(data_array.size/2) #calculates size
        input_array = np.zeros(size) #initalises the array
        if self.one_input: #if data is single input
            input_array = self.make_single_array(data_array,0,size)#make the array
            input_array = np.reshape(input_array,(-1,1)) #reshape
        else:
            input_array = self.make_double_in_array(data_array) #make the array with 2 column input
        return input_array
        
    '''
    returns array with function output
    '''
    def expected_output_array(self):
        data_array = self.pandas_to_numpy(self.pandas_data)#turns the read input dataframe ot numpy array
        one_in_size = int(data_array.size/2)
        two_in_size = 100
        if self.one_input:
            output_array = self.make_single_array(data_array, 1, one_in_size)
            output_array = np.reshape(output_array, (-1,1))
        else:
            output_array = self.make_single_array(data_array, 2, two_in_size)
            output_array = np.reshape(output_array, (-1,1))
        return output_array
  