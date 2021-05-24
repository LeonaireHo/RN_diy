from Module import *

class Sigmoide(Module):
    def __init__(self):
        pass

    def forward(self, X):
        X = np.maximum(X,1e-5)
        return 1/(1+np.exp(-X))

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        # return self.forward(input)*(1-self.forward(input))
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # print("sig",input.shape,delta.shape)
        return (self.forward(input)*(1-self.forward(input))) * delta
    def zero_grad(self):
        ## Annule gradient
        pass

class Tanh(Module):
    def __init__(self):
        pass

    def forward(self, X):
        ## Calcule la passe forward
        # print("TanhForward",X,np.tanh(X))
        return np.tanh(X)

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        # return (1-self.forward(input)*2)@delta
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        # print("tanh", input.shape, delta.shape)
        return (1-self.forward(input)**2)*delta
    def zero_grad(self):
        ## Annule gradient
        pass

class ReLu(Module):
    def __init__(self):
        pass

    def forward(self, X):
        ## Calcule la passe forward
        return X * (X > 0)

    def update_parameters(self, gradient_step=1e-3):
        ## Calcule la mise a jour des parametres selon le gradient calcule et le pas de gradient_step
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        return (input > 0) * np.ones(input.shape) * delta
    def zero_grad(self):
        ## Annule gradient
        pass
