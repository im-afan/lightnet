#TODO: add batch training
#gradient calculators
import core, layers, activations, losses
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
        
        param_grads = [None for i in range(len(model.varsArr))]
        memo = np.zeros((out[-1].shape))

        for i in range(len(out[-1])):
            memo[i] = self.loss.getGrad(out[-1], y, i)

        for i in range(len(model.varsArr)-1, -1, -1):
            #print(i)
            memo, param_grads[i] = model.varsArr[i].backprop(memo, out_noactivation[i], out[i], out_noactivation[i+1], out[i+1])

        return param_grads
