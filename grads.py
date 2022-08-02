#gradient calculators
import core

class SchotasticGrad(core.AutoGrad):
    def __init__(self, loss):
        super(SchotasticGrad, self).__init__(loss)

    def getGrad(self, model):
        grad = [np.zeros(model[i].weights.shape) for i in range(len(model.weightsArr))]
        print(grad)
        #for i in range(len(model.weightsArr)-1, -1, -1):
            
