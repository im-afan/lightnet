#trains a neural network to find the average of two numbers
#this doesn't do anything different than the average 2 numbers example, just tests hiden layer backprop 

from lightnet import core, activations, grads, losses
import numpy as np

act = core.Activation(lambda x: x, lambda x: 1)

weight1 = core.FeedForward(np.random.uniform(size=(2, 2)), np.zeros((2,)), act) 
weight2 = core.FeedForward(np.random.uniform(size=(2, 1)), np.zeros((1,)), act)

model = core.Sequential([weight1, weight2])

model.trainableArr = [True, True]

print("test call: ",model.call(np.array([1, 1]), training=True))

sgd = grads.SchotasticGrad(losses.MSE())

out = model.call(np.array([1, 1]), training=True)

for layer in out:
    print(layer)


mse = losses.MSE()
lr = 0.1

print("intial weights: ")
print(model.varsArr[0].weights)
print(model.varsArr[1].weights)
print("\n")

for i in range(1000):
    a = np.random.uniform()
    b = np.random.uniform()

    gradients = sgd.getGrad(model, np.array([a, b]), np.array([(a+b)/2])) 
    
    for j in range(len(gradients)):
        model.varsArr[j].weights -= lr*gradients[j]

    if(i % 200 == 0):
        print("output: ", model.call(np.array([a, b])), "expected: ", np.array([(a+b)/2]))
    #print("input: ", a, b)
    #print("loss: ", mse.getLoss(model, np.array([a, b]), np.array([(a+b)/2])))

print("final weights: ")
print(model.varsArr[0].weights)
print(model.varsArr[1].weights)
