from Linear import *
from Non_linear import *
from Module import  *

class BCE(Loss):
    def bce_log(self,y):
        size = y.shape
        return np.array([[max(-100,np.log(y[i,j] + 1e-5)) for j in range(size[1])] for i in range(size[0])])

    def forward(self, y, yhat):
        return - (y*self.bce_log(yhat) + (1 - y) * self.bce_log(1 - yhat))

    def backward(self, y, yhat):
        return - (y / (yhat + 1e-5) - (1 - y) / ((1 - yhat) + 1e-5))

