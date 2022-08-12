from lightnet import core
import numpy as np

class Dense(core.FeedForward): #two completely connected layers; can be thought of as a nn with 0 hidden layers
    def __init__(self, weights, biases, activation): #TODO: add more customization (for flatten layers, etc.)
        super(Dense, self).__init__()
        self.weights = weights #output layer shape is determined by weights
        self.biases = biases
        self.activation = activation #class activation

    def call(self, inlayer):
        #return self.activation.func(inlayer @ self.weights + self.biases), inlayer @ self.weights + self.biases
        return self.activation.func(np.dot(inlayer, self.weights)+self.biases), np.dot(inlayer, self.weights)+self.biases
        