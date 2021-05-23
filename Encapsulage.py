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
        self.moduleList=moduleList #reaseau
        self.loss=loss() #fonction de cout
        self.eps=eps #un pas

    def step(self,batch_x,batch_y):
        #calcule la sortie du reseau
        self.moduleList.moduleList_update(batch_x,batch_y,self.loss,self.eps)

def SGD(moduleList,datax,datay,batch_size,ite):
    o=Optim(moduleList)
    for i in range(ite):
        inds=np.random.choice([i for i in range(len(datax))],size=batch_size)
        batch_x=datax[inds]
        batch_y=datay[inds]
        o.step(batch_x, batch_y)
    return moduleList