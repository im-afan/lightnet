import pandas as pd
import numpy as np
from lightnet import core, layers, grads, losses, activations

np.random.seed(1)

data = pd.read_csv("train.csv")

train_x = data.drop("label", axis=1).to_numpy()/255
labels = data["label"].to_numpy()
train_y = np.zeros((train_x.shape[0], 10))
for i in range(train_x.shape[0]):
    train_y[i][labels[i]] = 1 
#np.expand_dims(train_y, 1)

print(train_x.shape, train_y.shape)

inds = np.arange(train_x.shape[0])

weight1 = layers.Dense(np.random.uniform(size=(784, 10))-0.5, np.random.uniform(size=(10,))-0.5, activations.LeakyRelu())
#weight2 = layers.Dense(np.random.uniform(size=(10, 10))-0.5, np.random.uniform(size=(10,))-0.5, activations.Tanh())
weight3 = layers.Dense(np.random.uniform(size=(10, 10))-0.5, np.random.uniform(size=(10,))-0.5, activations.Sigmoid())
#weight3 = layers.Dense(np.random.uniform(size=(10, 10)), np.random.uniform(size=(10,)), activations.Tanh())
#weight4 = layers.Dense(np.random.uniform(size=(10, 3)), np.random.uniform(size=(3,)), activations.Sigmoid())

model = core.Sequential([weight1, weight3])

print(train_x[0].shape)

print(model.call(train_x[0]))

lr = 0.1

loss = losses.MSE()
grad = grads.SchotasticGrad(loss)

lossLog = []

for i in range(1):
    np.random.shuffle(inds)
    l = 0
    for j in range(train_x.shape[0]):
        a = inds[j]
        x = train_x[a]
        y = train_y[a]

        grads = grad.getGrad(model, x, y)
        
        l = loss.getLoss(model, x, y)

        for i in range(len(model.varsArr)):
            model.varsArr[i].apply_grads(grads[i], lr=lr)

        if(j % 100 == 0):
            k = np.random.randint(low=0, high=42000)
            print("out: ", model.call(train_x[k]), "expected: ", train_y[k])

        #print(l)
