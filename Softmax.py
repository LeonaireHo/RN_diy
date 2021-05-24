from Module import *
class Softmax(Module):
    def __init__(self):
        pass
    def forward(self, X):
        # print(X)
        # print("soft",(np.exp(X)/ np.sum(np.exp(X),axis = 1).reshape(X.shape[0],1)))
        X = np.maximum(X,1e-8)
        return np.exp(X)/ np.sum(np.exp(X),axis = 1).reshape(X.shape[0],1)

    def update_parameters(self,gradient_step=None):
        pass

    def backward_update_gradient(self, input=None, delta=None):
        pass

    def backward_delta(self,input,delta):
        return (self.forward(input)*(1  - self.forward(input))) * delta

class CEloss(Loss):
    def forward(self, y, yhat):
        # print("celoss",yhat.shape,y.shape)
        return np.array([-yhat[i,y[i]] for i in range(len(y))])

    def backward(self, y, yhat):
        res=np.zeros(yhat.shape)
        for i in range(len(yhat)):
            res[1,y[i]]= -1
        return res

class CrossEntropy(Loss):
    def forward(self, y, yhat):
        # print("cross",y.shape,y[1])
        return CEloss().forward(y, yhat)+ np.log(np.sum(np.exp(yhat),axis=1).reshape(yhat.shape[0],1))
    def backward(self, y, yhat):
        return CEloss().backward(y, yhat) + np.exp(yhat) / np.sum(np.exp(yhat),axis=1).reshape(yhat.shape[0],1)
