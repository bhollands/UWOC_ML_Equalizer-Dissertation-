Author - Bernard Hollands

Oringally written for F20BC at Heriot-Watt and adapted for final year project

This is a Feed forward Multi-Layer Perceptron neural network optimised using 
Particel Swarm Optimisation for solving mathematical functions with the option of changing the activation function.

In current configuration all data can be seen in the 'Data' folder. All given data has been converted to .csv

once run this results are recored in 'NN_output_data' folder in the form .xlsx (excel files)

---Requirments---
Python3 (3.8.6) - Older versions may work
Numpy (1.18.5)
Pandas (1.1.3)
openpyxl (3.0.5)
scipy (1.5.2)

---To run projcect--- 
Please run main.py.
You can adjust parameters all parameters from the 'singleRun()' function 

---Choise of Data---
Sine wave - "sine"
Linear Function - "linear"
Hyperbolic Tangent - "tanh"
Complex Function - "complex"
XOR Funcion - "xor"
UWOC Data - "uwoc" : Dissertation Specific

---Choice of activation functions---
Cosine - "cosine"
Hyperbolic Tangent - "hypertangent"
Gaussian function - "gaussian"
Sigmoid Function - "sigmoid"

You can select either any of the excel files in the <Data> directory 
or equ to run on the 600Mpbs dataset.

singlerun() - This simply trains the network for the specified settings. The network output and fitness are recorded.
In the singleRun files the columns are as follows for "_results_<file>_<activation function>"
0 - Actual output from the function
1 - Network output for that run

For "_fitness_results_<file>_<activation function>"
 0 - fitness per epoch

