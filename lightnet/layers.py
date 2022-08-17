from lightnet import core, activations
import numpy as np

def load_layer(path, name):
    #for i in range(len(name_list)):
    layer_config = open(path+"/" + name + "_config", "r")
    layer_type = layer_config.readline()
    if(layer_type == "dense"):
        weights = np.load(path+"/" + name + "_weights.npy")
        biases = np.load(path+"/" + name + "_biases.npy")
        return Dense(weights, biases, activations.Linear())

class Dense(core.FeedForward): #two completely connected layers; can be thought of as a nn with 0 hidden layers
    def __init__(self, weights, biases, activation, name="dense"): #TODO: add more customization (for flatten layers, etc.)
        super(Dense, self).__init__()
        self.weights = weights #output layer shape is determined by weights
        self.biases = biases
        self.activation = activation #class activation
        self.name = name

    def call(self, inlayer):
        #return self.activation.func(inlayer @ self.weights + self.biases), inlayer @ self.weights + self.biases

        return self.activation.func(np.dot(inlayer, self.weights)+self.biases), np.dot(inlayer, self.weights)+self.biases
    
    def backprop(self, memo_activations, z1, a1, z2, a2):
        #memo_activations: backprop memoization
        #z is output w/o activation function of next layer
        #a is output from previous layer
        #new_memo = np.zeros((self.weights.shape[0]))
        """
        for j in range(len(new_memo)): 
            new_memo[j] = np.dot(memo_activations * self.activation.grad(z2), self.weights[j])
        """
        new_memo = np.dot(memo_activations * self.activation.grad(z2), self.weights.T)

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

    def save_layer(self, dir):
        np.save(dir + "/" + self.name + "_weights", self.weights)
        np.save(dir + "/" + self.name + "_biases", self.biases)
        config_file = open(dir + "/" + self.name + "_config", "w")
        config_file.write("dense") #only stores layer type for now
        

class Conv2D(core.FeedForward):
    def __init__(self, filters, activation):
        super(Conv2D, self).__init__()
        self.activation = activation
        self.filters = filters
        self.filter_shape = filters[0].shape

    def call(self, inlayer): #CHANNELS LAST INPUT
        print(self.filter_shape)
        outlayer = np.zeros((inlayer.shape[0]-self.filter_shape[0], inlayer.shape[0]-self.filter_shape[1], len(self.filters)))
        for newchannel in range(len(self.filters)):
            for i in range(len(inlayer)-self.filter_shape[0]):
                for j in range(len(inlayer[i])-self.filter_shape[1]):
                    print(i, j)
                    mult = self.filters[newchannel] * inlayer[i:i+self.filter_shape[0], j:j+self.filter_shape[1]]
                    outlayer[i][j][newchannel] = np.sum(mult)

        return self.activation.func(outlayer), outlayer

    def backprop(self, memo_activations, z1, a1, z2, a2): #TODO
        grads = [np.zeros(self.filter_shape) for i in range(len(self.filters))]
        newmemo = np.zeros(a1.shape)

        for grad in range(len(grads)):
            for i in range(len(newmemo)):
                for j in range(len(newmemo[i])):
                    pass



    def apply_grads(self, grads): #grads: one for each filter
        for i in range(len(self.filters)):
            self.filters[i] -= grads[i]

class Flatten(core.FeedForward):
    def __init__(self):
        super(Flatten, self).__init__()

    def backprop(self, memo_activations, z1, a1, z2, a2):
        return memo_activations.reshape(a1.shape)

    def call(self, inlayer):
        return inlayer.flatten()
