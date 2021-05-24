from Module import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from Linear import Linear, MSE
from Encapsulage import Sequential,Optim,SGD
from Softmax import Softmax,CEloss,CrossEntropy
import Non_linear
from AutoEncoder import *
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
        res_mse = mse.forward(datay, res_lin)
        delta_mse = mse.backward(datay,res_lin)
        l_mse.append(sum(res_mse))
        linear.zero_grad()
        linear.backward_update_gradient(datax,delta_mse)
        linear.update_parameters()
    return l_mse,"Linear",100

def testNonLinear(X,Y):
    sigmoide = Non_linear.Sigmoide()
    tanh = Non_linear.Tanh()
    coche1 = Linear(2,5)
    coche2 = Linear(5,1)
    mse = MSE()
    res_sigmoide = None
    loss = []
    maxIter = 100
    for _ in range(maxIter):
        #forward
        res_lin1 = coche1.forward(X)
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
        coche2.zero_grad()
        coche2.backward_update_gradient(res_tanh, delta_sig)
        coche2.update_parameters(0.05)

        delta_lin2 = coche2.backward_delta(X, delta_sig)
        delta_tanh = tanh.backward_delta(res_lin1,delta_lin2)
        coche1.zero_grad()
        coche1.backward_update_gradient(X, delta_tanh)
        coche1.update_parameters(0.05)
    return loss,"Nonlinear",maxIter

def testSequential(X,Y):
    #construct
    seq = Sequential()
    seq.add_module(Linear(2,5))
    seq.add_module(Non_linear.Tanh())
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
    seq.add_module(Non_linear.Tanh())
    seq.add_module(Linear(5,1))
    seq.add_module(Non_linear.Sigmoide())
    def fctSig(res):
        return np.array([[1 if res[i] > 0.5 else 0] for i in range(len(res))])
    #evolute
    maxIter = 100
    optim = Optim(seq,fctsort = fctSig)
    for _ in range(maxIter):
        optim.step(X,Y)
    return optim.moduleList.histLoss,"Optim",maxIter

def testSGD(X,Y):
    #construct
    seq = Sequential()
    seq.add_module(Linear(2,5))
    seq.add_module(Non_linear.Tanh())
    seq.add_module(Linear(5,1))
    seq.add_module(Non_linear.Sigmoide())
    def fctSig(res):
        return np.array([[1 if res[i] > 0.5 else 0] for i in range(len(res))])
    #evolute
    maxIter = 300
    rn = SGD(seq,X,Y,50,MSE,fctSig,maxIter)
    return rn.moduleList.histLoss,"SGD",maxIter


def load_usps(fn):
    with open(fn,"r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def testSoftmax():
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    X, Y = load_usps(uspsdatatrain)
    Xtest, Ytest = load_usps(uspsdatatest)
    onehot = np.zeros((Y.size, 10),dtype=np.int)
    onehot[np.arange(Y.size), Y] = 1
    # print(X.shape,Y.shape,onehot.shape)
    seq = Sequential()
    seq.add_module(Linear(256, 50))
    seq.add_module(Non_linear.Tanh())
    seq.add_module(Linear(50, 10))
    seq.add_module(Softmax())
    # print(X[0])
    # return 0

    # evolute
    maxIter = 100
    optim = Optim(seq,loss=CrossEntropy,eps=0.01)
    # print(onehot[0])
    for _ in range(maxIter):
        optim.step(X, onehot)
    return optim.moduleList.histLoss, "Sodtmax", maxIter

def testAutoEncoder():
    #pepre data
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    alltrainx, alltrainy = load_usps(uspsdatatrain)
    alltestx, alltesty = load_usps(uspsdatatest)
    neg = 9
    pos = 6
    datax, datay = get_usps([neg, pos], alltrainx, alltrainy)
    datay = np.array([1 if datay[i] == 6 else 0 for i in range(len(datay))])
    testx, testy = get_usps([neg, pos], alltestx, alltesty)
    maxIter = 100
    #rn encodage
    encodage = Sequential()
    encodage.add_module(Linear(256, 100))
    encodage.add_module(Non_linear.Tanh())
    encodage.add_module(Linear(100, 10))
    encodage.add_module(Non_linear.Tanh())
    # rn decodage
    encodage.add_module(Linear(10, 100))
    encodage.add_module(Non_linear.Tanh())
    encodage.add_module(Linear(100, 256))
    encodage.add_module(Non_linear.Sigmoide())
    #rn decodage
    # decodage = Sequential()
    # decodage.add_module(Linear(10, 100))
    # decodage.add_module(Non_linear.Tanh())
    # decodage.add_module(Linear(100, 256))
    # decodage.add_module(Non_linear.Sigmoide())
    for i in range(maxIter):
        #forward
        # print(datax[0])
        print(i)
        encodage.forward(datax)
        # print(encodage.forwards[-1][0])
        encodage.backward(datax,datax,loss=BCEloss,gradient_step=0.1)
        if i % 10 == 0:
            # plt.figure()
            # plt.imshow(datax[0].reshape(16, 16), cmap="gray")
            # plt.title("Image original de 9: {}".format(datay[0]))
            # plt.savefig("plot/num/origine9.png")
            # plt.close()
            plt.figure()
            plt.imshow(encodage.forwards[-1][-10].reshape(16, 16), cmap="gray")
            plt.title("Image apres autoEncoder de 6".format(datay[0]))
            plt.savefig("plot/num/6_iter"+i.__str__()+".png")
            plt.close()
    return encodage.histLoss,"AutoEncoder",maxIter

#test**************************************
#init data
datax = np.random.randn(100,2)
datay = np.array([1 if datax[i][1] > np.sin(datax[i][0]) else 0 for i in range(100)])

# datay = np.random.choice([-1,1],100,replace=True)
# # datay = np.array([1 if datax[i,0]*datax[i,1]>0 else 0 for i in range(100)])
#
# print(datay)
# dataymulti = np.random.choice(range(10),20,replace=True)

# loss,titre,ite = testlinear(datax,datay)
# loss,titre,ite = testNonLinear(datax,datay)
# loss,titre,ite = testSequential(datax,datay)
# loss,titre,ite = testOptim(datax,datay)
# loss,titre,ite = testSGD(datax,datay)
# loss,titre,ite = testSoftmax()
loss,titre,ite = testAutoEncoder()




# print(loss)
# plt.plot([i for i in range(0,ite)],loss,color="red",linewidth="3")
# plt.title(titre)
# plt.xlabel("iteration")
# plt.ylabel("loss")
# plt.savefig("plot/"+titre)

