from Module import *

class MSE(Loss):
    def forward(self, y, yhat):
        return np.linalg.norm((y.reshape(-1, 1)-yhat),axis=1)*2
    def backward(self, y, yhat):
        return 2 * (y.reshape((-1,1)) - yhat)

class Linear(Module):
    def __init__(self,input,output):
        self._parameters = np.random.uniform(-1,1,(input,output))
        self._gradient = np.zeros((input,output))
        self._sizein = input
        self._sizeout = input
        self.loss = MSE()
    def zero_grad(self):
        ## Annule gradient
        # print("grad",self._gradient.shape)
        self._gradient = np.zeros(self._gradient.shape)
        # print("grad",self._gradient)

    def forward(self, data):
        return data@self._parameters

    def backward_update_gradient(self, input, delta):
        # print("gard",self._parameters.shape,input[0],delta[0])
        self._gradient = self._gradient + input.T@delta
        # if self._parameters.shape == (256,100):
        #     print("grad",self._gradient[0][:5])

    def backward_delta(self, input, delta):
        #return self.loss.backward(self.forward(input),delta)
        # print("linear", input.shape, delta.shape)
        return delta @ self._parameters.T

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        # print("para",self._parameters.shape)
        # print(self._gradient)
        self._parameters += gradient_step*self._gradient
        if self._parameters.shape == (50, 10):
            print("param",self._parameters[0][:10])