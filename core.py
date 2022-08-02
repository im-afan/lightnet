import numpy as np

class FeedForward: #two completely connected layers; can be thought of as a nn with 0 hidden layers
    def __init__(self, weights, activation):
        self.weights = weights #output layer shape is determined by weights
        self.activation = activation

    def call(self, inlayer):
        return self.activation(inlayer * self.weights)
        
class Sequential: #links many FeedForwards into a completely connected ANN
    def __init__(self, weightsArr):
        self.weightsArr = weightsArr
        
    def call(self, inlayer, training=False):
        ret = []
        for i in range(len(self.weightsArr)):
            ret.append(inlayer)
            inlayer = self.weightsArr[i].call(inlayer)
        ret.append(inlayer)

        if(training):
            return inlayer
        else:
            return ret

#chain rule: dy/dx = dy/du * du/dx
#y is loss, x is weight -> u is 

#backprop step for a hidden layer of distance one from output layer: dC/dw = dC/da2 * da2/dw = dC/da2 * do(a1w)/a1w * da1w/dw
#= dC/da2 * da2/dw * d[o(x)]/dx * a1
#only one nn call needed for backprop

class Loss: #includes both gradient and loss function
    def __init__(self, loss, grad):
        self.loss = loss
        self.grad = grad #gradient with respect to output layer (output layer as input)

    def getLoss(self, model, x, y): #x = logit, y = label
        return self.loss(model.call(x), y)

    def getGrad(self, x):
        return self.grad(x)

class AutoGrad:
    def __init__(self, loss):
        self.loss = loss #class Loss type (sorry for potential confusion)
    
    def getGrad(self, model):
        pass

class TrainSession: #optimizer
    def __init__(self, model, loss, autograd):
        self.model = model
        self.loss = loss
        self.autograd = autograd #autograd calculates gradient step for each train step

    def trainBatch(self, logits, labels):
        for i in range(len(logits)):
            logit = logits[i]
            label = labels[i]
        