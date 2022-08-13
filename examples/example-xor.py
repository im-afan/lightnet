#ANN learns pattern for XOR
from lightnet import core, grads, losses, activations, layers
import numpy as np
from numpy.random import uniform
from random import randint
import matplotlib.pyplot as plt

np.random.seed(0)

#layers = []


w1 = layers.Dense(uniform(size=(2, 2))-0.5, uniform(size=(2,))-0.5, activations.Relu())
#w3 = core.FeedForward(uniform(size=(8, 8)), uniform(size=(8,)), activations.Sigmoid())
w2 = layers.Dense(uniform(size=(2, 1))-0.5, uniform(size=(1,))-0.5, activations.Sigmoid())


#w1 = core.FeedForward(np.array([[20, -20], [20, -20]]), np.array([-10, 30]), activations.Sigmoid())
#print(w1.weights.shape, w1.biases.shape)
#w2 = core.FeedForward(np.array([[20], [20]]), np.array([-30]), activations.Sigmoid())
#print(w2.weights.shape, w2.biases.shape)


model = core.Sequential([w1, w2])

train_x = np.array([np.array([1, 1]), np.array([0, 1]), np.array([0,0]), np.array([1,0])])
train_y = np.array([np.array([0]), np.array([1]), np.array([0]), np.array([1])])

loss = losses.MSE()
grad = grads.SchotasticGrad(losses.MSE())

grad.getGrad(model, np.array([1, 1]), np.array([0]))

lr = 0.1

print(grad.getGrad(model, train_x[0], train_y[0]))

losses = []

for i in range(10000):
    #print("epoch: ", i)
    l = 0
    r = np.random.randint(0, 3)
    for j in range(len(train_x)):
        #j = np.random.randint(0, 3)
        a = (j+r)%4
        x = train_x[a]
        y = train_y[a]

        grads = grad.getGrad(model, x, y)

        l += loss.getLoss(model, x, y)

        #"""
        if(i % 20 == 0):
            """
            print("(1, 1): ", model.call(np.array([1, 1])))
            print("(0, 1): ", model.call(np.array([0, 1])))
            print("(1, 0): ", model.call(np.array([1, 0])))
            print("(0, 0): ", model.call(np.array([0, 0])))
            """
            print(model.call(train_x))
            print("\n")   
        #"""
        for k in range(len(grads)):
            model.varsArr[k].weights -= lr * grads[k][0]
            model.varsArr[k].biases -= lr * grads[k][1]

    #print("loss:", l/4)
    losses.append(l)

print("final params:")
print("weights:", [w.weights for w in model.varsArr])
print("biases:", [w.biases for w in model.varsArr])

print("(1, 1): ", model.call(np.array([1, 1]), training=True))
print("(0, 1): ", model.call(np.array([0, 1]), training=True))
print("(1, 0): ", model.call(np.array([1, 0]), training=True))
print("(0, 0): ", model.call(np.array([0, 0]), training=True))

plt.plot(losses)
plt.show()