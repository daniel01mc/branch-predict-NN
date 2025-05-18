import os
import pandas as pd
import numpy as np
import utils
from predictors import (
        MLP,
        CNN,
        RNN
        )


# change here for other trace files 
trace = utils.read_data('./traces/perl_trace.txt')

results = {}
plot = False
normalize = True

#%% Best RNN model
print('_______________________________________________ RNN ___________________________________________________')
predictor = RNN(
    history=4,
    num_hidden_layers=3,
    num_units=32,
    activation='tanh',
    skip=True
    )
# load weights from best model
predictor.load(os.path.join('best_models',
                            'RNN_history_4_hidden_layers_3_num_units_32_skip_True_activation_tanh-perl',
                            'weights_23_0.6884.h5'))
predictor.model.summary()
y_pred, y_true = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(y_true, y_pred, name='Recurrent Neural Network',
       plot=True, normalize=normalize)
print('\n', results[predictor.name])




#%% Best MLP model
print('_______________________________________________ MLP ___________________________________________________')
predictor = MLP(
        history=4,
        num_hidden_layers=9,
        neurons_per_layer=32,
        activation='relu',
        )
predictor.load(os.path.join('best_models',
                            'MLP_history_4_hidden_layers_3_neurons_32_activation_relu-perl',
                            'weights_25_0.6882.h5'))
predictor.model.summary()

y_pred, y_true = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(y_true, y_pred, name='Multilayer MLP',
       plot=True, normalize=normalize)
print('\n', results[predictor.name])

#%% Output to csv
df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('algorithms_results.csv')



#%% Best CNN model
print('_______________________________________________ CNN ___________________________________________________')
predictor = CNN(
    history=4,
    num_hidden_layers=9,
    num_filters=32,
    kernel_size=3,
    activation='relu',
    dilation=1,
    skip=True
    )
predictor.load(os.path.join('best_models',
                            'CNN_history_4_hidden_layers_3_num_filters_32_kernel_size_3_skip_True_dilation_1_activation_relu-perl',
                            'weights_13_0.6867.h5'))
predictor.model.summary()

y_pred, y_true = predictor.predict(trace['Branch'])
results[predictor.name] = utils.evaluate(y_true, y_pred, name='Convolutional Neural Network',
       plot=True, normalize=normalize)
print('\n', results[predictor.name])

 
# CNN grid search
paths = [p for p in os.listdir('logs') if 'CNN' in p]

for path in paths:
    weights = [f for f in os.listdir(os.path.join('logs', path)) if f.endswith('.h5')][0]
    print(os.path.join('logs', path, weights))
 
    history = int(path.split('_')[2])
    
    predictor = CNN(
        history=history,
        num_hidden_layers=3,
        num_filters=32,
        kernel_size=3,
        activation='relu',
        dilation=1,
        skip=True
        )
    predictor.load(os.path.join('logs', path, weights))
    
    predictor.name = path
    y_pred, y_true = predictor.predict(trace['Branch'])
    results[predictor.name] = utils.evaluate(y_true, y_pred, name='Convolutional Neural Network',
           plot=False, normalize=normalize)
    
    results[predictor.name]['history'] = int(path.split('_')[2])
    results[predictor.name]['layers'] = int(path.split('_')[5])
    results[predictor.name]['dilation'] = int(path.split('_')[15])
    results[predictor.name]['activation'] = path.split('_')[-1]
    results[predictor.name]['parameters'] = predictor.model.count_params()
    print('\n', results[predictor.name])


df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('cnn_results.csv')

# Best MLP model
paths = [p for p in os.listdir('logs') if 'MLP' in p]

for path in paths:
    weights = [f for f in os.listdir(os.path.join('logs', path)) if f.endswith('.h5')][0]
    print(os.path.join('logs', path, weights))
    
    history = int(path.split('_')[2])
    
    predictor = MLP(
            history=history,
            num_hidden_layers=3,
            neurons_per_layer=32,
            activation='relu'
            )
    predictor.load(os.path.join('logs', path, weights))
    predictor.name = path
    y_pred, y_true = predictor.predict(trace['Branch'])
    results[predictor.name] = utils.evaluate(y_true, y_pred, name='Multilayer MLP',
           plot=False, normalize=normalize)
    results[predictor.name]['history'] = int(path.split('_')[2])
    results[predictor.name]['layers'] = int(path.split('_')[5])
    results[predictor.name]['activation'] = path.split('_')[-1]
    results[predictor.name]['parameters'] = predictor.model.count_params()
    print('\n', results[predictor.name])

df = pd.DataFrame.from_dict(results, orient='index')
df.to_csv('mlp_results.csv')
