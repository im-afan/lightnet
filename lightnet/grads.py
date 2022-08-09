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
        #print(memo_activations)
        """
        partDerivSum_out = 0 #sum of all partial derivatives of cost w/ respect to output layer
        memo = 1

        for i in range(len(model.varsArr)):
            for ind in np.ndindex(model.varsArr[i].weights.shape):
                grad = memo_activations[i+1][ind[1]]
                grad *= model.varsArr[i].activation.grad(out_noactivation[i+1][ind[1]])
                grad *= out[i][ind[0]]
                grad_weights[i][ind[0]][ind[1]] = grad
            for ind in np.ndindex(model.varsArr[i].biases.shape):
                grad = memo_activations[i+1][ind[0]]
                grad *= model.varsArr[i].activation.grad(out_noactivation[i+1][ind[0]])
                grad_biases[i][ind[0]] = grad
        """
        """
        for i in range(len(model.varsArr)-1, -1, -1):
            temp = 0
            #print("i: ", i)
            mult = 0 #multiply to memo
            #if(not model.trainableArr[i]):
            #    continue

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
                    #print("memo: ", memo)
                    add = model.varsArr[i].activation.grad(out_noactivation[i+1][ind[1]])
                    #print(add * model.varsArr[i].weights[ind[0]][ind[1]])
                    #print(type(model.varsArr[i].weights[ind[0]][ind[1]]))
                    #print(type(add))
                    add = add * model.varsArr[i].weights[ind[0]][ind[1]]
                    dw = 1
                    dw *= memo #problem is here
                    dw *= model.varsArr[i].activation.grad((out_noactivation[i+1][ind[1]]))
                    dw *= out[i][ind[0]]
                    mult += add
                    grad_weights[i][ind[0]][ind[1]] = dw
            
            memo *= mult 
            
            for ind in np.ndindex(model.varsArr[i].biases.shape): #memo is updated in weights sgd - backprop for biases is pretty much the same (only last part different)
                db = 1
                db *= memo
                db *= out_noactivation[i+1][ind[0]]
                grad_biases[i][ind[0]] = db
            
        """

        return grad_weights, grad_biases

