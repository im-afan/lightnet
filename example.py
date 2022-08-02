import core
import losses
import activations
import grads
import numpy as np

weight1 = core.FeedForward(np.array([[np.random.uniform()]]), lambda x: x)
weight2 = core.FeedForward(np.array([[np.random.uniform()]]), lambda x: x)

model = core.Sequential([weight1, weight2])

sgd = grads.SchotasticGrad(losses.MSE)

print(model.call(np.array([1])))

