import core
import numpy as np

class MSE(core.Loss):
    def __init__(self):
        def meanSquaredError(y, yhat):
            return 1/2 * (np.mean(y-yhat))**2

        def meanSquaredErrorGrad(y, yhat):
            n = len(y)
            grad = []
            for i in range(n):
                grad.append((y[i]-yhat[i])/n)

            return np.array(grad)

        super(MSE, self).__init__(meanSquaredError, meanSquaredErrorGrad)
