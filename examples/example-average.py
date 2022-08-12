#trains a neural network to find the average of two numbers
from lightnet import core, grads, losses, activations
import numpy as np

act = core.Activation(lambda x: x, lambda x: 1) #linear activation function

weight1 = core.FeedForward(np.random.uniform(size=(2, 1)), np.zeros((1,)), act) #initialize random weights
model = core.Sequential([weight1]) #create 

sgd = grads.SchotasticGrad(losses.MSE())

out = model.call(np.array([1, 1]), training=True)

for layer in out:
    print(layer)


mse = losses.MSE()
lr = 0.1

for i in range(1000):
    a = np.random.uniform()
    b = np.random.uniform()

    gradients, biasgrads = sgd.getGrad(model, np.array([a, b]), np.array([(a+b)/2])) 
    
    for j in range(len(gradients)):
        model.varsArr[j].weights -= lr*gradients[j]

    print("pred: ", model.call(np.array([a, b]))[0], " expected: ", (a+b)/2)

    print("input: ", a, b)
    print("loss: ", mse.getLoss(model, np.array([a, b]), np.array([(a+b)/2])))