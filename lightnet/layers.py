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
    
    def backprop(self, memo_activations, z1, a1, z2, a2):
        #memo_activations: backprop memoization
        #z is output w/o activation function of next layer
        #a is output from previous layer
        new_memo = np.zeros((self.weights.shape[0]))
        for j in range(len(new_memo)): 
            new_memo[j] = np.dot(memo_activations * self.activation.grad(z2), self.weights[j])

        grad_weights = np.zeros(self.weights.shape)
        grad_biases = np.zeros(self.biases.shape)

        for j in range(len(new_memo)):
            for k in range(len(memo_activations)):
                grad_weights[j][k] = memo_activations[k] * self.activation.grad(z2[k]) * a1[j]

        for j in range(len(memo_activations)):
            grad_biases[j] = memo_activations[j] * self.activation.grad(z2[j])
        #returns: backprop memo array, gradients for this layer's weights
        
        return new_memo, [grad_weights, grad_biases]

    def apply_grads(self, grads, lr=0.1):
        self.weights -= lr * grads[0]
        self.biases -= lr * grads[1]