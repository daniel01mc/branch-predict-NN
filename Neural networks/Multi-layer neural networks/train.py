from predictors import CNN, Perceptron, RNN
import utils


trace = utils.read_data('./traces/jpeg_trace.txt')

epochs = 25
batch_size = 256


predictor = RNN(
    history=21,
    num_hidden_layers=3,
    num_units=32,
    activation='tanh',
    skip=True
    )

"""
predictor = CNN(
    history=21,
    num_hidden_layers=3,
    num_filters=32,
    kernel_size=3,
    activation='relu',
    dilation=1,
    skip=True
    )


predictor = Perceptron(
    history=21,
    num_hidden_layers=3,
    neurons_per_layer=32,
    activation='relu'
    )
""" 

predictor.fit(trace['Branch'], epochs=epochs, batch_size=batch_size)