import numpy as np
from lightnet import core
from math import exp, tanh

class Relu(core.Activation):
    def __init__(self):
        """
        def reluScalar(x):
            return max(0, x)
        def reluScalarGrad(x):
            return x > 0
        """
        super(Relu, self).__init__(lambda z: np.maximum(z, 0), lambda z: z > 0)
        
class LeakyRelu(core.Activation):
    def __init__(self, a=0.2):
        def leakyReluScalar(x):
            return max(x*a, x)
        def leakyReluScalarGrad(x):
            if(x > 0):
                return 1
            return a
        super(LeakyRelu, self).__init__(lambda arr: np.where(arr>0, arr, arr*0.1), lambda arr: np.where(arr>0, 1, 0.1))

class Sigmoid(core.Activation):
    def __init__(self):
        def sigmoidScalar(x):
            return 1/(1+exp(-x))
        def sigmoidScalarGrad(x):
            return 1/(1+exp(-x)) * (1-(1/(1+exp(-x))))

        super(Sigmoid, self).__init__(np.vectorize(sigmoidScalar), np.vectorize(sigmoidScalarGrad))

class Linear(core.Activation):
    def __init__(self):
        def linearScalar(x):
            return x
        def linearScalarGrad(x):
            return 1

        super(Linear, self).__init__(np.vectorize(linearScalar), np.vectorize(linearScalarGrad))

class Tanh(core.Activation):
    def __init__(self):
        def tanhScalar(x):
            return tanh(x)
        def tanhScalarGrad(x):
            return 1-tanh(x)**2
        
        super(Tanh, self).__init__(np.vectorize(tanhScalar), np.vectorize(tanhScalarGrad))
"""
relu = np.vectorize(reluScalar)
sigmoid = np.vectorize(reluScalar)
linear = np.vectorize(linearScalar)
"""