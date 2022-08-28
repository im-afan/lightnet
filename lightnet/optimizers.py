from lightnet import core, layers, grads, losses, activations

class SchotasticGradSession(core.TrainSession):
    def __init__(self, model, loss, autograd):
        super(SchotasticGradSession, self).__init__(model, loss, autograd)

    def trainStep(self, x, y):
        grads = self.autograd.getGrad(self.model, x, y)
        for i in range(len(grads)):
            self.model.varsArr[i].apply_grads(grads[i])
        
    def fitModel(self, train_x, train_y, epochs=1):
        for i in range(epochs):
            for j in range(len(train_x)):
                self.trainStep(train_x[j], train_y[j])