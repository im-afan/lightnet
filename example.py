import core
import losses
import activations
import grads
import numpy as np

weight1 = core.FeedForward(np.ones((1,2)), lambda x: x)
weight2 = core.FeedForward(np.ones((2,1)), lambda x: x)

model = core.Sequential([weight1, weight2])

#sgd = grads.SchotasticGrad(losses.MSE)

out = model.call(np.array([1]), training=True)

for layer in out:
    print(layer)

