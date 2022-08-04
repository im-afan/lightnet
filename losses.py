import core
import numpy as np

class MSE(core.Loss):
    def __init__(self):
        def meanSquaredError(y, yhat):
            return 1/2 * (np.mean(y-yhat))**2

        def meanSquaredErrorGrad(y, yhat, i): #only for 1d output for now
            n = y.size
            return 1/n * (y[i]-yhat[i])

        super(MSE, self).__init__(meanSquaredError, meanSquaredErrorGrad)



        
