from Module import *

class MSE(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm((y-yhat),axis=1)*2
    def backward(self, y, yhat):
        return 2 * (y - yhat)

class Linear(Module):
    def __init__(self,input,output):
        self._parameters = np.random.random((input,output))
        self._gradient = np.zeros((input,output))
        self._sizein = input
        self._sizeout = input
        self.loss = MSE()

    def forward(self, data):
        return data@self._parameters

    def backward_update_gradient(self, input, delta):
        # print("grad",input.shape,delta.shape)
        self._gradient = input.T@delta

    def backward_delta(self, input, delta):
        #return self.loss.backward(self.forward(input),delta)
        return delta @ self._parameters.T

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        self._parameters += gradient_step*self._gradient