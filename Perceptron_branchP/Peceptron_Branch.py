import sys
import os
from collections import deque
import time
import numpy as np
import pandas as pd
import codecs
import math
import sys
import argparse

class Perceptron:
    weights = []
    N = 0
    bias = 0
    threshold = 0
    
    # Initialize
    def __init__(self, N):
        self.N = N
        self.bias = 0
        # Threshold depends on history length
        self.threshold = 2 * N + 21
        self.weights = [0] * N
    
    # Prediction function
    def predict(self, global_branch_history):
        summation = self.bias
    
        # Compute the dot product weights with branch historty sum =  x_i * w_i
        for i in range(0, self.N):
        
            summation += global_branch_history[i] * self.weights[i]
            
        # If the prediction output is less than the output of the dot product prediction = -1. +1 otherwise
        prediction = -1 if summation < 0 else 1
        
        # return prediction and summation
        return (prediction, summation)

    # Update model function
    def update(self, prediction, actual, global_branch_history, summation):
        
        #Adjust the weights
        if (prediction != actual) or (abs(summation) < self.threshold):
            self.bias = self.bias + (1 * actual)
            
            for i in range(0, self.N):
                self.weights[i] = self.weights[i] + (actual * global_branch_history[i])

# Perceptron # of correct and incorrect values prediction function
def peceptron_branch_correct(trace, l=4, tablesize=None):
    global_branch_history = deque([])
    global_branch_history.extend([0] * l)

    prediction_list = {}
    correct_pred = 0
    incorrect = 0

    # iterating branches and hash table size
    for br in trace:
        if tablesize:
            index = hash(br[0]) % tablesize
        else:
            index = hash(br[0])

        # Check for previous branches
        if index not in prediction_list:
            prediction_list[index] = Perceptron(l)
            
        # Get results from prediction
        results = prediction_list[index].predict(global_branch_history)
        
        prev_r = results[0]
        summation = results[1]
        
        # Convert values in bipolar integers
        # If the current value of branch is 1 get 1. -1 Otherwise
        
        actual_value = 1 if br[1] else -1
                
        # Update model
        prediction_list[index].update(prev_r, actual_value, global_branch_history, summation)
        global_branch_history.appendleft(actual_value)
        global_branch_history.pop()
        
        # if the previous outcome is equal to the actual value the correct number of predictions increase
        # else if the previous outcome is not equal to the actual value the incorrect number of predictions increase
        if prev_r == actual_value:
            correct_pred += 1
        else:
            incorrect += 1
    return correct_pred, incorrect, len(prediction_list)


# Read Arg as string
trace = sys.argv[1]
# Read file as string
ndata = np.loadtxt(trace, dtype='str')

# Adapt them into a DataFrame
df = pd.DataFrame(ndata, columns = ['A','B'])

# Translate address into decimal and convert 0 and 1 for T/N
df['B'] = df['B'].map({'n': 0 ,'t': 1}).astype(int)
df['A'] = df['A'].apply(int, base =16)

# Ouput the correct file convertion to use in Perceptron
with open('perceptron_trace.txt', 'w') as f:
    df_string = df.to_string(header=False, index=False)
    f.write(df_string)

# Get converted file recently created in the same directory
inputFile = 'perceptron_trace.txt'

#inputFile = codecs.open(trace_filename,'r','utf8')
#print(inputFile)

with open(inputFile, 'r') as branchpfile:
    trace = []
    for line in branchpfile.readlines():
        tok = line.split(' ')
        trace.append([tok[0], int(tok[1])])

correct_pred, incorrect, num_p = peceptron_branch_correct(trace, 21)
print('\nLength of Perceptrons list: ' + str(num_p) + '\n') 

result_acc= []
result_time = []
for i in range (1, 3):
    
    # Start Time
    start_time = time.time()
    
    # Returns number of correct predictions and lenght of list
    correct_pred, incorrect, num_p  = peceptron_branch_correct(trace, i)
    # End time
    end_time = time.time()
    # Calculate miss prediction rate
    result_acc.append((correct_pred/float(len(trace))))
    result_time.append((end_time - start_time) * 10**3 )
    
    #Display number of iterations
    print('iteration:' + str(i))
    print('Accuracy',result_acc[-1])
    print('Time execution in ms', result_time[-1])
    print('\n')

# Display information
print('\nTotal number of predictions: ' + str((len(trace))))
print('Incorrect Predictions: ' + str(incorrect))
print('Correct Predictions: ' + str(correct_pred))
print('Miss Prediction Rate: ' + str(incorrect/float(len(trace))*100) + '\n')
print('\nModel Accuracy: ' + str(correct_pred/float(len(trace))*100) + '\n')

# Save Model Accuracy
file_out = open('results.csv', 'a')
file_out.write(str(incorrect/float(len(trace))*100) + '\n')
file_out.close()

