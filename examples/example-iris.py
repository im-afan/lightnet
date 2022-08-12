import pandas as pd
import numpy as np
from lightnet import core, grads, activations, losses, layers
import matplotlib.pyplot as plt

weight1 = layers.Dense(np.random.uniform(size=(4, 10))-0.5, np.random.uniform(size=(10,))-0.5, activations.Relu())
weight2 = layers.Dense(np.random.uniform(size=(10, 10))-0.5, np.random.uniform(size=(10,))-0.5, activations.Relu())
#weight3 = core.FeedForward(np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,)), activations.Tanh())
weight4 = layers.Dense(np.random.uniform(size=(10, 3))-0.5, np.random.uniform(size=(3,))-0.5, activations.Sigmoid())

np.random.seed(1)

model = core.Sequential([weight1, weight2, weight4])

dataframe = pd.read_csv("iris.csv")

labels = dataframe["variety"]
train_y = []
for i in range(len(labels)):
    if(labels[i] == "Setosa"):
        train_y.append([1, 0, 0])
    if(labels[i] == "Versicolor"):
        train_y.append([0, 1, 0])
    if(labels[i] == "Virginica"):
        train_y.append([0, 0, 1])
train_y = np.array(train_y)

inds = np.arange(150)
np.random.shuffle(inds)

input_ = dataframe.drop("variety", 1)
train_x = input_.to_numpy()
max_x = np.zeros((train_x.shape[1]))

test_ind = 100

for i in range(len(train_x)):
    for j in range(len(train_x[i])):
        max_x[j] = max(max_x[j], train_x[i][j])

for i in range(len(train_x)):
    for j in range(len(train_x[j])):
        train_x[i][j] = train_x[i][j] / max_x[j]

lr = 0.1
loss = losses.MSE()
grad = grads.SchotasticGrad(loss)

losses = []

print(len(model.call(train_x)))

for i in range(200):
    l = 0
    
    for j in range(test_ind):
        
        if(i < 120):
            lr = 1
        else:
            lr = 0.1
        a = inds[j]
        x = train_x[a]
        y = train_y[a]

        weightGrads, biasGrads = grad.getGrad(model, x, y)

        l += loss.getLoss(model, x, y)

        for k in range(len(weightGrads)):
            model.varsArr[k].weights -= lr * weightGrads[k]
            model.varsArr[k].biases -= lr * biasGrads[k]
    print("epoch: ", i+1)
    print("loss:", l)
    losses.append(l)

print(model.varsArr[0].weights, model.varsArr[1].weights, model.varsArr[2].weights)
print(model.varsArr[0].biases, model.varsArr[1].biases, model.varsArr[2].biases)

plt.plot(losses)
plt.show()

print(model.call(train_x))

for i in range(test_ind, 150):
    print("out: ", model.call(train_x[inds[i]]))
    print("expected: ",  train_y[inds[i]])
