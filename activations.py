import numpy as np
from math import exp

class Relu(core.Activation):
    def __init__(self):
        def reluScalar(x):
            return max(0, x)
        def reluScalarGrad(x):
            if(x > 0):
                return 1
            return 0

        super(Relu, self).__init__(np.vectorize(reluScalar), np.vectorize(reluScalarGrad))
        
class Sigmoid(core.Activation):
    def __init__(self):
        def sigmoidScalar(x):
            return 1/(1+exp(-x))
        def sigmoidScalarGrad(x)
            return 1/(1+exp(-x)) * (1-(1/(1+exp(-x))))

        super(Sigmoid, self).__init__(np.vectorize(sigmoidScalar), np.vectorize(sigmoidScalarGrad))

class Linear(core.Activation):
    def __init__(self):
        def linearScalar(x):
            return x
        def linearScalarGrad(x):
            return 1

        super(Linear, self).__init__(np.vectorize(linearScalar), np.vectorize(linearScalarGrad))

relu = np.vectorize(reluScalar)
sigmoid = np.vectorize(reluScalar)
linear = np.vectorize(linearScalar)