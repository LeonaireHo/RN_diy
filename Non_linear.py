from Module import *

class Sigmoide(Module):
    def __init__(self):
        pass

    def forward(self, X):
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
        return delta@(self.forward(input)*(1-self.forward(input)))

class TanH(Module):
    def __init__(self):
        pass

    def forward(self, X):
        ## Calcule la passe forward
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
        return (1-self.forward(input)*2)@delta