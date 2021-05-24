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
            # print(self.forwards[-1])
            self.forwards.append(self.moduleList[i].forward(self.forwards[-1]))
        return self.forwards[-1]

    def backward(self,X,Y,gradient_step=1e-3,fctsort = None,loss = MSE):
        #loss
        loss = loss()
        res = self.forwards[-1]
        if fctsort:
            res = fctsort(res)
        # print(Y.shape,res.shape)
        self.histLoss.append(sum(loss.forward(Y, res)))
        # retro-propager
        delta = loss.backward(Y, res)
        # print("delta",delta[0])
        for i in range(len(self.moduleList)-1,0,-1):
            self.moduleList[i].zero_grad()
            self.moduleList[i].backward_update_gradient(self.forwards[i-1], delta)
            self.moduleList[i].update_parameters(gradient_step)
            delta = self.moduleList[i].backward_delta(self.forwards[i-1], delta)
        self.moduleList[0].zero_grad()
        self.moduleList[0].backward_update_gradient(X, delta)
        self.moduleList[0].update_parameters(gradient_step)

class Optim:
    def __init__(self,moduleList,loss=MSE,eps=1e-3,fctsort = None):
        self.moduleList=moduleList
        self.loss=loss
        self.eps=eps
        self.fctsort = fctsort

    def step(self,batch_x,batch_y):
        # print("step",self.moduleList.forward(batch_x)[0],batch_y[0])
        self.moduleList.backward(batch_x, batch_y,fctsort = self.fctsort,gradient_step=self.eps,loss = self.loss)

def SGD(moduleList,X,Y,batch_size,loss,fctsort,maxiter):
    rn = Optim(moduleList,loss=loss,fctsort=fctsort)
    for _ in range(maxiter):
        indx=np.random.choice([i for i in range(len(X))],size=batch_size)
        batch_x=X[indx]
        batch_y=Y[indx]
        rn.step(batch_x, batch_y)
    return rn