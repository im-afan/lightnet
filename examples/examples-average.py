#trains a neural network to find the average of two numbers

import core
import losses
import activations
import grads
import numpy as np

act = core.Activation(lambda x: x, lambda x: 1)

weight1 = core.FeedForward(np.random.uniform(size=(2, 1)), np.zeros((2,)), act) 

model = core.Sequential([weight1])

sgd = grads.SchotasticGrad(losses.MSE())

out = model.call(np.array([1, 1]), training=True)

for layer in out:
    print(layer)


mse = losses.MSE()
lr = 0.1

for i in range(1000):
    a = np.random.uniform()
    b = np.random.uniform()

    gradients = sgd.getGrad(model, np.array([a, b]), np.array([(a+b)/2])) 
    
    for j in range(len(gradients)):
        model.varsArr[j].weights -= lr*gradients[j]

    print("input: ", a, b)
    print("loss: ", mse.getLoss(model, np.array([a, b]), np.array([(a+b)/2])))
