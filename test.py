from torch.nn import Tanh

from Module import *
import numpy as np
import matplotlib.pyplot as plt
from Linear import Linear, MSE
from Encapsulage import Sequential,Optim,SGD
import Non_linear

# np.random.seed(1)

def testlinear(datax,datay):
    ## Lineaire et MSE
    linear = Linear(10,1)
    linear.zero_grad()
    mse = MSE()
    # print(datax,datay)
    l_mse = []
    for _ in range(100):
        res_lin = linear.forward(datax)
        res_mse = mse.forward(datay.reshape(-1,1), res_lin)
        delta_mse = mse.backward(datay.reshape(-1,1),res_lin)
        l_mse.append(sum(res_mse))
        linear.backward_update_gradient(datax,delta_mse)
        linear.update_parameters()
    return l_mse,"Linear",100

def testNonLinear(X,Y):
    sigmoide = Non_linear.Sigmoide()
    tanh = Non_linear.TanH()
    coche1 = Linear(2,2)
    coche2 = Linear(2,1)
    mse = MSE()
    res_sigmoide = None
    loss = []
    maxIter = 100
    for _ in range(maxIter):
        #forward
        res_lin1 = coche1.forward(datax)
        res_tanh = tanh.forward(res_lin1)
        res_lin2 = coche2.forward(res_tanh)
        res_sigmoide = sigmoide.forward(res_lin2)
        #loss
        res = np.array([[1 if res_sigmoide[i] > 0.5 else 0] for i in range(len(res_sigmoide))])
        # res = np.array([res_sigmoide[i] > 0 for i in range(len(res_sigmoide))])
        # print("res",res.shape)
        loss.append(sum(mse.forward(Y.reshape(-1, 1), res)))
        #retro-propager
        res_mse = mse.backward(Y.reshape(-1, 1), res)
        # print("mse",res_mse.shape)
        delta_sig = sigmoide.backward_delta(res_lin2,res_mse)
        coche2.backward_update_gradient(res_tanh, delta_sig)
        coche2.update_parameters(0.05)

        delta_lin2 = coche2.backward_delta(datax, delta_sig)
        delta_tanh = tanh.backward_delta(res_lin1,delta_lin2)
        coche1.backward_update_gradient(datax, delta_tanh)
        coche1.update_parameters(0.05)
    return loss,"Nonlinear",maxIter

def testSequential(X,Y):
    #construct
    seq = Sequential()
    seq.add_module(Linear(2,5))
    seq.add_module(Non_linear.TanH())
    seq.add_module(Linear(5,1))
    seq.add_module(Non_linear.Sigmoide())
    def fctSig(res):
        return np.array([[1 if res[i] > 0.5 else 0] for i in range(len(res))])
    #evolute
    maxIter = 100
    for _ in range(maxIter):
        seq.forward(X)
        seq.backward(X,Y,fctsort = fctSig)
    return seq.histLoss,"Sequential",maxIter

def testOptim(X,Y):
    #construct
    seq = Sequential()
    seq.add_module(Linear(2,5))
    seq.add_module(Non_linear.TanH())
    seq.add_module(Linear(5,1))
    seq.add_module(Non_linear.Sigmoide())
    def fctSig(res):
        return np.array([[1 if res[i] > 0.5 else 0] for i in range(len(res))])
    #evolute
    maxIter = 100
    optim = Optim(seq,loss = fctSig)
    for _ in range(maxIter):
        optim.step(X,Y)
    return optim.moduleList.histLoss,"Optim",maxIter

def testSGD(X,Y):
    #construct
    seq = Sequential()
    seq.add_module(Linear(2,5))
    seq.add_module(Non_linear.TanH())
    seq.add_module(Linear(5,1))
    seq.add_module(Non_linear.Sigmoide())
    def fctSig(res):
        return np.array([[1 if res[i] > 0.5 else 0] for i in range(len(res))])
    #evolute
    maxIter = 300
    rn = SGD(seq,X,Y,50,fctSig,maxIter)
    return rn.moduleList.histLoss,"SGD",maxIter

#init data
datax = np.random.randn(100,2)
datay = np.random.choice([-1,1],100,replace=True)
# datay = np.array([1 if datax[i,0]*datax[i,1]>0 else 0 for i in range(100)])
datay = np.array([1 if datax[i][1] > np.sin(datax[i][0]) else 0 for i in range(100)])

print(datay)
# dataymulti = np.random.choice(range(10),20,replace=True)

#loss,titre,ite = testlinear(datax,datay)
# loss,titre,ite = testNonLinear(datax,datay)
# loss,titre,ite = testSequential(datax,datay)
# loss,titre,ite = testOptim(datax,datay)
loss,titre,ite = testSGD(datax,datay)
print(loss)
plt.plot([i for i in range(0,ite)],loss,color="red",linewidth="3")
plt.title(titre)
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig("plot/"+titre)