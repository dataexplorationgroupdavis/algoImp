# python3

# Backpropagation Algorithm from scratch
# Modified from an online script
# source from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from random import random, seed
from math import exp
import numpy as np

# initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    '''
    only one hidden layer now
    n_inputs: number of features
    n_hidden: number of nuerons in a hidder layer
    n_outputs: number of outputs

    each neuron is stored as a dictionary
    '''
    network = list()
    hidden_layer = [{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)] # +1 for bias term
    network.append(hidden_layer)
    output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    '''
    calculation of activation, like a regresion
    linear combination: w1*x1 + w2*x2 + ... 
    '''
    activation = weights[-1] # bias term
    for i in range(len(weights)-1):
        activation += weights[i] * inputs[i]
    return activation

# Transfer neuron activation
def transfer(activation):
    '''
    from linear to non-linear
    sigmoid/logtistic regression 
    '''
    return 1.0 / (1.0 + exp(-activation))
    # value = 1.0 / (1.0 + exp(-activation))
    # if(value < 0.5):
    #     return 0
    # return 1

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
    return np.array(inputs)

# Calculate the derivative of an neuron output
def transfer_derivative(output):
    # derivative of logistic function here
    return output * (1.0 - output)

# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    '''
    network: network
    expected: real output 
    delta: the error signal
    '''
    for i in reversed(range(len(network))):
        layer = network[i]
        errors = list()
        if i != len(network) - 1:
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected - neuron['output'])
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
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


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
        for i in range(len(Xtrain)):
            row = Xtrain[i]
            outputs = forward_propagate(network, row)
            # print('output:',outputs)
            expected = ytrain[i] 
            #  sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            sum_error += (expected - outputs) ** 2
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))

def predict(network, row):
    '''
    wait to be done
    '''

def run_nn(Xtrain, ytrian, Xtest, ytest, l_rate, n_epoch, n_hidden):
    '''
    wait to be done
    '''
    n_inputs = len(Xtrain[0])
    n_outputs = len(set(ytrain))
    network = initialize_network(n_inputs, n_hidden, n_outputs)
    train_network(network, Xtrain, ytrain, l_rate, n_epoch, n_outputs)
    predictions = []
    for i in len(range(Xtest)):
        prediction = predict(network, Xtest[i])
        predictions.append(prediction)
    return predictions


# TODO
# testing
def main():
    from mnist import MNIST
    mndata = MNIST("../MNIST/samples")
    Xtrain, ytrain = mndata.load_training()
    Xtest, ytest = mndata.load_testing()
    #  zerosAndOnes = ytrain == 0 or ytrain == 1
    #  X = Xtrain[zerosAndOnes]
    #  y = ytrain[zerosAndOnes]
    #  zerosAndOnes2 = ytest == 0 or ytest == 0
    #  Xt = Xtest[zerosAndOnes2]
    #  yt = ytest[zerosAndOnes2]
    #  n_inputs = len(X[0])
    network = initialize_network(n_inputs=len(Xtrain[0]), n_hidden=10, n_outputs=1)
    import pdb
    #  pdb.set_trace()
    train_network(network, Xtrain, ytrain, l_rate = 0.1, n_epoch = 6)

if __name__ == '__main__':
    main()
