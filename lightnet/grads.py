#gradient calculators
from lightnet import core, layers, activations, losses
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
        
        grad_weights = [np.zeros(model.varsArr[i].weights.shape) for i in range(len(model.varsArr))]
        grad_biases = [np.zeros(model.varsArr[i].biases.shape) for i in range(len(model.varsArr))] #TODO: gradients for biases

        memo_activations = [np.zeros(out[i].shape) for i in range(len(out))]

        for i in range(len(memo_activations)-1, 0, -1):
            if(i == len(memo_activations)-1):
                for j in range(len(memo_activations[i])):
                    memo_activations[i][j] = self.loss.grad(out[-1], y, j) #correct
            else:
                for j in range(len(memo_activations[i])): #TODO; optimize this 
                    dsum = 0
                    for k in range(len(memo_activations[i+1])):
                        dsum += memo_activations[i+1][k] * model.varsArr[i].activation.grad(out_noactivation[i+1][k]) * model.varsArr[i].weights[j][k]
                    #print(dsum)
                    memo_activations[i][j] = dsum

        
            for j in range(len(memo_activations[i-1])):
                for k in range(len(memo_activations[i])):
                    grad_weights[i-1][j][k] = memo_activations[i][k] * model.varsArr[i-1].activation.grad(out_noactivation[i][k]) * out[i-1][j]

            for j in range(len(memo_activations[i])):
                grad_biases[i-1][j] = memo_activations[i][j] * model.varsArr[i-1].activation.grad(out_noactivation[i][j])

        return grad_weights, grad_biases

