from Module import *

class Sigmoide(Module):
    def forward(self, X):
        ## Calcule la passe forward
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass

class TanH(Module):
    def __init__(self):
        self._parameters = None
        self._gradient = None

    def forward(self, X):
        ## Calcule la passe forward
        pass

    def backward_update_gradient(self, input, delta):
        ## Met a jour la valeur du gradient
        pass

    def backward_delta(self, input, delta):
        ## Calcul la derivee de l'erreur
        pass