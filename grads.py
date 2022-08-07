#gradient calculators
from lightnet import core
import numpy as np

#chain rule: dy/dx = dy/du * du/dx
#y is loss, x is weight -> u is 

#backprop step for a hidden layer of distance one from output layer: dC/dw = dC/da2 * da2/dw = dC/da2 * do(a1w)/a1w * da1w/dw
#= dC/da2 * da2/dw * d[o(x)]/dx * a1
#only one nn call needed for backprop


class SchotasticGrad(core.AutoGrad):
    def __init__(self, loss):
        super(SchotasticGrad, self).__init__(loss)

    def getGrad(self, model, x, y):
        out, out_noactivation = model.call(x, training=True)
        
        #print("out: ", out[-1][0], " expected: ", y[0])

        #loss = self.loss.loss(out, y)
        weightProd = 1

        grad_weights = [np.zeros(model.varsArr[i].weights.shape) for i in range(len(model.varsArr))]
        grad_biases = [] #TODO: gradients for biases

        partDerivSum_out = 0 #sum of all partial derivatives of cost w/ respect to output layer
        memo = 1

        for i in range(len(model.varsArr)-1, -1, -1):
            temp = 0
            #print("i: ", i)
            mult = 0 #multiply to memo
            if(not model.trainableArr[i]):
                continue
            for ind in np.ndindex(model.varsArr[i].weights.shape):
                #print(ind[0], ind[1])
                if(i == len(model.varsArr)-1):
                    dw = 1
                    dw *= self.loss.grad(out[-1], y, ind[1])
                    mult += dw #for memo
                    dw *= model.varsArr[i].activation.grad((out_noactivation[i+1][ind[1]]))
                    dw *= out[i][ind[0]]
                    grad_weights[i][ind[0]][ind[1]] = dw
                else:
                    print("memo: ", memo)
                    add = model.varsArr[i].activation.grad((out_noactivation[i+1][ind[1]]))
                    add *= model.varsArr[i].weights[ind[0]][ind[1]]
                    dw = 1
                    dw *= memo
                    dw *= model.varsArr[i].activation.grad((out_noactivation[i+1][ind[1]]))
                    dw *= out[i][ind[0]]
                    mult += add
                    grad_weights[i][ind[0]][ind[1]] = dw
            
            memo *= mult    


        return grad_weights