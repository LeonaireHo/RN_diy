from Linear import  *
class Sequential:
    def __init__(self):
        self.moduleList=[]
        self.forwards = None
        self.histLoss = []

    def add_module(self,module):
        self.moduleList.append(module)

    def forward(self,datax):
        self.forwards=[self.moduleList[0].forward(datax)]
        for i in range(1,len(self.moduleList)):
            self.forwards.append(self.moduleList[i].forward(self.forwards[-1]))
        return self.forwards[-1]

    def backward(self,X,Y,gradient_step=1e-3,fctsort = None):
        #loss
        mse = MSE()
        res = self.forwards[-1]
        if fctsort:
            res = fctsort(res)
        self.histLoss.append(sum(mse.forward(Y.reshape(-1, 1), res)))
        # retro-propager

        delta = mse.backward(Y.reshape(-1, 1), res)
        for i in range(len(self.moduleList)-2,0,-1):
            self.moduleList[i].backward_update_gradient(self.forwards[i-1], delta)
            self.moduleList[i].update_parameters(gradient_step)
            delta = self.moduleList[i].backward_delta(self.forwards[i-1], delta)
        self.moduleList[0].backward_update_gradient(X, delta)
        self.moduleList[0].update_parameters(gradient_step)

class Optim:
    def __init__(self,moduleList,loss=MSE,eps=1e-3):
        self.moduleList=moduleList
        self.loss=loss
        self.eps=eps

    def step(self,batch_x,batch_y):
        self.moduleList.forward(batch_x)
        self.moduleList.backward(batch_x, batch_y,fctsort = self.loss,gradient_step=self.eps)

def SGD(moduleList,X,Y,batch_size,loss,maxiter):
    rn = Optim(moduleList,loss=loss)
    for _ in range(maxiter):
        indx=np.random.choice([i for i in range(len(X))],size=batch_size)
        batch_x=X[indx]
        batch_y=Y[indx]
        print(batch_x,batch_y)
        rn.step(batch_x, batch_y)
    return rn