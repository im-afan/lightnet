#gradient calculators
import core
import numpy as np

#chain rule: dy/dx = dy/du * du/dx
#y is loss, x is weight -> u is 

#backprop step for a hidden layer of distance one from output layer: dC/dw = dC/da2 * da2/dw = dC/da2 * do(a1w)/a1w * da1w/dw
#= dC/da2 * da2/dw * d[o(x)]/dx * a1
#only one nn call needed for backprop


class SchotasticGrad(core.AutoGrad):
    def __init__(self, loss):
        super(SchotasticGrad, self).__init__(loss)

    #def getWeightGrad(self, out, y, layer, i, j):
        #deriv = self.loss.getGrad(out[0], y[0], 0) 
        #for i in range(layer-1):


    def getGrad(self, model, x, y):
        out, out_noactivation = model.call(x, training=True)
        
        #loss = self.loss.loss(out, y)
        weightProd = 1

        grad_weights = [np.zeros(model.varsArr[i].weights.shape) for i in range(len(model.varsArr))]
        grad_biases = []

        for i in range(len(model.varsArr)-1, -1, -1):
            temp = 0
            for ind in np.ndindex(model.varsArr[i].weights.shape):
                if(i == len(model.varsArr)-1):
                    dw = 1
                    print("output: ", out[-1][0], " expected: ", y[0])
                    dw *= self.loss.grad(out[-1], y, ind[1])
                    #print(model.varsArr[i].weights, out[i])
                    dw *= model.varsArr[i].activation.grad((out_noactivation[i+1])) #TODO: add pre-activation feature for sequential api
                    dw *= out[i][ind[0]]
                    grad_weights[i][ind[0]][ind[1]] = dw

        return grad_weights