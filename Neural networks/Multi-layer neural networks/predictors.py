import os
import numpy as np
import keras.backend as K
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model, Sequential
from keras.layers import Input, Dense, Conv1D, Add, Flatten, SimpleRNN
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint


states = {
            1: {
                'T': {
                    'prediction': 1, # predicts taken
                    'transition': (
                            'N',    # incorrect prediction: transition to NT state
                            'T'      # correction prediction: transition to T state
                            )},
                'N': {
                    'prediction': 0, # predicts not taken
                    'transition': (
                            'T',     # incorrect prediction: transition to T state
                            'N'     # correction prediction: transition to NT state
                            )}
                    },
                    
            2: {
                'ST': {
                    'prediction': 1, # predicts taken
                    'transition': (
                            'WT',    # incorrect prediction: transition to WT state
                            'ST'     # correction prediction: transition to ST state
                            )},
                'WT': {
                    'prediction': 1, # predicts taken
                    'transition': (
                            'WN',   # incorrect prediction: transition to WN state
                            'ST'     # correction prediction: transition to ST state
                            )},
                'SN': {
                    'prediction': 0, # predicts not taken
                    'transition': (
                            'WN',   # incorrect prediction: transition to WN state
                            'SN'    # correction prediction: transition to SN state
                            )},
                'WN': {
                    'prediction': 0, # predicts not taken
                    'transition': (
                            'WT',    # incorrect prediction: transition to WT state
                            'SN'    # correction prediction: transition to SN state
                            )}
                }
        }
                    
                    
class Predictor(object):

    def predict(self, y_true):
        raise NotImplementedError


class NeuralNetwork(Predictor):

    def __build_model(self):
        raise NotImplementedError
        
        
    def load(self, filename):              
        self.model = load_model(filename)
        
        
    def _preprocess(self, y_true):
	# define X, y based on history parameter and create train-test data

        X = np.zeros((len(y_true)-self.history, self.history), dtype='uint8')
        y = np.zeros((len(y_true)-self.history), dtype='uint8')
        
        for i, j in enumerate(range(self.history, len(y_true))):
            X[i, :] = y_true[j-self.history:j]
            y[i] = y_true[j]

        if len(self.input_shape) > 1:
            X = np.expand_dims(X, axis=-1)
            
        return train_test_split(X, y, test_size=0.3, shuffle=False, random_state=0)
        
    
    def fit(self, y_true, epochs=50, batch_size=64, tb=False):
	# train the model and make training logs

        X_train, X_test, y_train, y_test = self._preprocess(deepcopy(y_true))
        
  
        if not os.path.exists(os.path.join('logs', self.name)):
            os.mkdir(os.path.join('logs', self.name))

        callbacks = []
        callbacks.append(CSVLogger(filename=os.path.join('logs', self.name, 'train_log.csv')))
        
        callbacks.append(ModelCheckpoint(filepath=os.path.join('logs', self.name,
                                    'weights_{epoch:02d}_{val_loss:.4f}.h5'),
                                    save_best_only=True, save_weights_only=False))

        if tb:
            callbacks.append(TensorBoard(log_dir=os.path.join('logs', self.name, 'tb'),
                            histogram_freq=1, write_graph=False, write_images=False))
        
      
        _ =  self.model.fit(X_train, y_train, validation_data=(X_test, y_test),
                            epochs=epochs, batch_size=batch_size, callbacks=callbacks)


    def predict(self, y_true):
        """ Predict on test set """
        
        _, X_test, _, y_test = self._preprocess(deepcopy(y_true))
        
        y_pred = self.model.predict(X_test).squeeze().round().astype(int)
        
        return list(y_pred), list(y_test)
    
    
class MLP(NeuralNetwork):


    def __init__(self,
                 history,
                 num_hidden_layers=3,
                 neurons_per_layer=32,
                 activation='relu'
                 ):
        
        self.history = history
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_layer = neurons_per_layer
        self.activation = activation
        
        self.name = '_'.join(['MLP',
                              'history',
                              str(self.history),
                              'hidden_layers',
                              str(self.num_hidden_layers),
                              'neurons',
                              str(self.neurons_per_layer),
                              'activation',
                              self.activation
                              ])

        self.input_shape = (self.history,)
        self.model = self.__build_model()


    def __build_model(self):
        K.clear_session()

        inputs = Input(shape=self.input_shape)
        x = inputs
	
	# define MLP architecture with forward and backward prop

        for _ in range(self.num_hidden_layers):
            x = Dense(units=self.neurons_per_layer, activation=self.activation)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(lr=1E-3), loss='binary_crossentropy')
        return model



class RNN(NeuralNetwork):

    def __init__(self,
                 history,
                 num_hidden_layers=3,
                 num_units=32,
                 activation='tanh',
                 skip=False):
        self.history = history
        self.num_hidden_layers = num_hidden_layers
        self.num_units = num_units
        self.activation = activation
        self.skip = skip

        self.name = '_'.join(['RNN',
                              'history',
                              str(self.history),
                              'hidden_layers',
                              str(self.num_hidden_layers),
                              'num_units',
                              str(self.num_units),
                              'skip',
                              str(self.skip),
                              'activation',
                              self.activation
                              ])

        self.input_shape = (self.history, 1)
        self.model = self.__build_model()

    def __build_model(self):
        K.clear_session()

        inputs = Input(shape=self.input_shape)
        x = inputs
	
	# define RNN architecture with forward and backward prop

        for _ in range(self.num_hidden_layers):
            shortcut = x
            x = SimpleRNN(units=self.num_units, activation=self.activation, return_sequences=True)(x)
            
            if self.skip:
                x = Add()([x, shortcut])

        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(), loss='binary_crossentropy')
        return model



class CNN(NeuralNetwork):
    
    def __init__(self,
                 history,
                 num_hidden_layers=3,
                 num_filters=32,
                 kernel_size=3,
                 skip=False,
                 dilation=1,
                 padding='same',
                 activation='relu',
                 ):
        self.history = history
        self.num_hidden_layers = num_hidden_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.skip = skip
        self.dilation = dilation
        self.strides = 1
        self.padding = 'same'
        
        self.name = '_'.join(['CNN',
                              'history',
                              str(self.history),
                              'hidden_layers',
                              str(self.num_hidden_layers),
                              'num_filters',
                              str(self.num_filters),
                              'kernel_size',
                              str(self.kernel_size),
                              'skip',
                              str(self.skip),
                              'dilation',
                              str(self.dilation),
                              'activation',
                              self.activation
                              ])

        self.input_shape = (self.history, 1)
        self.model = self.__build_model()


    def __build_model(self):
        K.clear_session()

        inputs = Input(shape=self.input_shape)
        x = inputs
	
	# define CNN architecture with forward and backward prop

        for _ in range(self.num_hidden_layers):
            shortcut = x
            x = Conv1D(filters=self.num_filters,
                       kernel_size=self.kernel_size,
                       padding=self.padding,
                       strides=self.strides,
                       dilation_rate=self.dilation,
                       activation=self.activation
                       )(x)
            
            if self.skip:
                x = Add()([x, shortcut])

        x = Flatten()(x)
        outputs = Dense(1, activation='sigmoid')(x)

        model = Model(inputs, outputs)
        model.compile(optimizer=Adam(), loss='binary_crossentropy')
        return model