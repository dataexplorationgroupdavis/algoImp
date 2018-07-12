# python3

# Artificial Neural Network from scratch
# test data: MNIST dataset
# Modified from an online script
# source from https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

'''
Workflow
- initialize_network 
- train_network
   - forward_propagate 
   - backward_propagate_error 
   - update_weights 
- predict


data structure
- network: list
  - layer: list
    - neuron: dictionary
      - weights: list
      - delta: list
      - outputs: list
'''
from random import random, seed
from math import exp, log
import numpy as np
import pdb
from time import time

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
    for j in range(len(n_hidden)):
        if j == 0:
            hidden_layer = [{'weights': np.random.rand(n_inputs+1)} for i in range(n_hidden[j])]
        else:
            hidden_layer = [{'weights': np.random.rand(n_hidden[j-1]+1)} for i in range(n_hidden[j])]
        network.append(hidden_layer)

    output_layer = [{'weights': np.random.rand(n_hidden[-1]+1)} for i in range(n_outputs)]
    network.append(output_layer)
    return network

# Calculate neuron activation for an input
def activate(weights, inputs):
    '''
    calculation of activation, like a regresion
    linear combination: w1*x1 + w2*x2 + ... 
    '''

    # normalization
    regression = weights.dot(inputs)
    if regression < 0:
        regression = -log(-regression)
    else:
        regression = log(regression)

    # sigmoid 
    activation = 1.0 / (1.0 + exp(-regression))

    # hyperbolic tangent
    # activation = (exp(regression) - exp(regression)) / (exp(regression) + exp(-regression))

    return activation

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
            # activation is a large number
            # need to be normalized to make algorithm more efficient
            activation = activate(neuron['weights'], np.append(inputs,1))
            #  neuron['output'] = transfer(activation)
            neuron['output'] = activation 
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
            #  for j in range(len(inputs)):
            #      neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            #  neuron['weights'][-1] += l_rate * neuron['delta']
            neuron['weights'] += l_rate * neuron['delta'] * np.append(inputs, 1)


# Train a network for a fixed number of epochs
def train_network(network, Xtrain, ytrain, l_rate, n_epoch):
    '''
    network: whole network
    Xtrain, ytrain: training data
    l_rate: learning rate
    n_epoch: number times run through network
    '''
    for epoch in range(n_epoch):
        starttime = time()
        sum_error = 0
        for i in range(len(Xtrain)):
            row = Xtrain[i]
            outputs = forward_propagate(network, row)
            #  print('output:',outputs)
            expected = np.zeros(len(outputs))
            expected[ytrain[i]] = 1
            sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
            #  sum_error += (expected - outputs) ** 2
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)
            #  pdb.set_trace()
        print('>epoch=%d, lrate=%.3f, error=%.3f, time=%d' % (epoch, l_rate, sum_error, time() - starttime))

def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.argmax() 

def main():
    from mnist import MNIST
    mndata = MNIST("../MNIST/samples")
    Xtrain, ytrain = mndata.load_training()
    Xtest, ytest = mndata.load_testing()

    # normalize data
    Xtrain = np.array(Xtrain) / 255.0
    Xtest = np.array(Xtest) / 255.0
    n_outputs = len(set(ytrain))
    n_inputs = len(Xtrain[0])
    network = initialize_network(n_inputs=n_inputs, n_hidden=[int((n_inputs+n_outputs)/2)], n_outputs=n_outputs)
    #  pdb.set_trace()
    train_network(network, Xtrain, ytrain, l_rate = 1, n_epoch = 16)
    predictions = []
    for row in Xtest:
        predictions.append(predict(network, row))
    print(sum(np.array(predictions) == np.array(ytest)) / float(len(ytest)))

if __name__ == '__main__':
    main()

# TODO
# 1. make algorithm work (check)
# 2. understand backpropagation error (check)
# 3. make the number of hidden layers arbitrary (check) (does not help much, and not sure why)
# 4. try to use np.array instead of list
#      function                  |    if works
#    - update weights            |    600s to 35s
#    - backpropagation error     |
#    - activation (done)         |    700s to 600s 
# 5. understand activation function and transfer function

# NOTE
# for each epoch, it takes around 700 seconds (see, if we can use
# numpy array to reduce time)
# n_hidden is about 400, accuracy is 0.977, each epoch is about 480s
# n_hidden is about 250, accuracy is 0.974, each epoch is about 300s
