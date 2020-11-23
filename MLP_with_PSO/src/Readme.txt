This project was developed for F20BC class at Heriot-Watt University by Bernard Hollands and Abbie Prescott

This is a Feed forward Multi-Layer Perceptron neural network optimised using 
Particel Swarm Optimisation for solving mathematical functions with the option of changing the activation function.

In current configuration all data can be seen in the 'Data' folder. All given data has been converted to .csv

once run this results are recored in 'NN_output_data' folder in the form .xlsx (excel files)

---Requirments---
Python3
Numpy (1.18.5)
Pandas (1.1.3)
openpyxl (3.0.5)

---To run projcect--- 
Please run main.py.
In this file you can run either 'singleRun' or 'recordAllPossible' by uncommenting it. You can also chnage the parameters there

recordAllPossible() - If uncommented this will run the network specified on every datafile for every activation function and output the results to be graphed so the most effective activation funcation can be seen for that configuration.
 For every Neural Network run the fitness (mse loss) can be seen in the terminal agenst the epoch

In the excel file the columns are as follows for the file "_results_<file>": 
0 - Actual output from the mathemetical function
1 - Sigmoid Activation
2 - Hypertangent Activation
3 - Cosine Activatin
4 - Guassian Activation


singlerun() - This simply trains the network for the specified settings. The network output and fitness are recorded.
In the singleRun files the columns are as follows for "_results_<file>_<activation function>"
0 - Actual output from the function
1 - Netowrk output for that run

For "_fitness_results_<file>_<activation function>"
 0 - fitness per epoch


---Run it on your Data--- 
This is fully possible if your data models a mathematical function (i.e one or two, in one out)

This program will work if the data is in .csv format a small modification to select() function in the data_read.py file will make it possible

A file pso_attempt 1 has also been included, this is a failed attempt at the pso algorithum
