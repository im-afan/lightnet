import core
import numpy as np

class MSE(core.Loss):
    def __init__(self):
        def meanSquaredError(y, yhat):
            s = 0
            for i in range(len(y)):
                s += (y[i]-yhat[i])**2
            
            return 1/(2*len(y)) * s

        def meanSquaredErrorGrad(y, yhat, i): #only for 1d output for now
            #print(y, yhat, i)
            return 1/len(y) * (y[i]-yhat[i])

        super(MSE, self).__init__(meanSquaredError, meanSquaredErrorGrad)



        
