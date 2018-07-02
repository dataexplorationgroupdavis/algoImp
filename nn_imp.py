# python3

# Backpropagation Algorithm from scratch
# neural network for binary classification
# Modified from an online script
# source from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from random import random, seed
from math import exp

# initialize a network
def initialize_network(n_inputs, n_hidden):
    '''
    n_inputs: number of features
    n_hidden: number of hidder layers
    '''
    network = list()
    hidden_layer = [{'weight':[random() for i in range(n_inputs)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weight':[random() for i in range(n_hidden)]} for i in range(2)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    '''
    linear combination: w1*x1 + w2*x2 + ... 
    '''
    activation = weights 
    for i in range(len(weights)):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    '''
    sigmoid/logtistic regression 
    '''
    value = 1.0 / (1.0 + exp(-activation))
    if(value < 0.5):
        return 0
    return 1

# Forward propagate input to a network output
def forward_propagate(network, row):
    '''
    usage: training model through hidden_layers 
    network: network 
    row: one row of training data 
    '''
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    # derivative of logistic function here
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    '''
    network: network
    expected: real output 
    '''
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in networks[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

# Update network weights with error
def update_weights(network, row, l_rate):
    '''
    row: a row of raw data
    l_rate: learning rate
    '''
    for i in range(len(network)):
        inputs = row
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i-1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'] += l_rate * neuron['delta']

# Train a network for a fixed number of epochs
def train_network(network, Xtrain, ytrain, l_rate, n_epoch):
    '''
    network: whole network
    Xtrain, ytrain: training data
    l_rate: learning rate
    n_epoch: number times run through network
    '''
    for epoch in range(n_epoch):
        sum_error = 0
        for row in Xtrain:
            outputs = forward_propagate(network, row)
            actual = ytrain 
            sum_error += sum([(actual[i] - outputs[i])**2 for i in range(len(expected))])
            backward_propagate_error(network, actual)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# TODO
# testing
