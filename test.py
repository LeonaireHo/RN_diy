from torch.nn import Tanh

from Module import *
import numpy as np
import matplotlib.pyplot as plt
from Linear import Linear, MSE
from Non_linear import Sigmoide

np.random.seed(1)

datax = np.random.randn(20,10)
datay = np.random.choice([-1,1],20,replace=True)
dataymulti = np.random.choice(range(10),20,replace=True)

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
        grad_lin = linear._gradient
        delta_lin = linear.backward_delta(datax,delta_mse)
    plt.plot([i for i in range(0,100)],l_mse,color="red",linewidth="3")
    plt.title("Linear")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.savefig("plot/linear")

def testNonLinear(X,Y):
    sigmoide = Sigmoide()
    tanh = Tanh()
    coche1 = Linear(10,5)
    coche2 = Linear(5,1)
    mse = MSE()
    res_sigmoide = None
    loss = []
    for _ in range(100):
        #forward
        res_lin1 = coche1.forward(datax)
        res_tanh = tanh.forward(res_lin1)
        res_lin2 = coche2.forward(res_tanh)
        res_sigmoide = sigmoide.forward(res_lin2)
        #loss
        res_mse2 = mse.forward(Y.reshape(-1, 1), res_sigmoide)
        res = [res_sigmoide[i] > 0 for i in range(res_sigmoide.size())]
        loss.append(mse.forward(Y.reshape(-1, 1), res))
        #retro-propager
        sigmoide.backward_delta()
        res_lin2.backward_update_gradient(res_tanh, res_mse2)
        res_lin2.update_parameters()
        delta_lin2 = res_lin2.backward_delta(datax, res_mse2)

        delta_tanh = tanh.backward_delta(res_lin1,delta_lin2)
        res_lin1.backward_update_gradient(datax, delta_tanh)


testlinear(datax,datay)