from Module import *

class Conv1D(Module):
    def __init__(self,k_size,chan_in,chan_out,stride):
        self._k_size = k_size
        self._chan_in = chan_in
        self._chan_out = chan_out
        self._stride = stride
        self._parameters = np.random.randn(self._k_size,self._chan_in,self._chan_out) * 0.1
        self._gradient = np.zeros((self._k_size,self._chan_in,self._chan_out))

    def zero_grad(self):
        self._gradient = np.zeros((self._k_size,self._chan_in,self._chan_out))

    def forward(self, X):
        size = X.shape
        res = np.zeros((size[0],(size[1] - self._k_size) // self._stride + 1,self._chan_out))
        size = res.shape

        for i in range(size[0]):
            for j in range(size[1]):
                for l in range(size[2]):
                    stride = j * self._stride
                    res[i,j,l] = np.sum(X[i][stride:stride + self._k_size] * self._parameters[:,:,l])
        return res

    def backward_update_gradient(self, input, delta):
        size = delta.shape
        for i in range(size[0]):
            for j in range(size[1]):
                for l in range(size[2]):
                    stride = j * self._stride
                    self._gradient[:,:,l] = input[i][stride :stride + self._k_size] * delta[i,j,l]

    def backward_delta(self, input, delta):
        res = np.zeros(input.shape)
        size = delta.shape
        for i in range(size[0]):
            for j in range(size[1]):
                for l in range(size[2]):
                    # print(i, j, l)
                    stride = j * self._stride
                    res[i][stride : stride + self._k_size] = self._parameters[:,:,l] * delta[i,j,l]
        return res

class MaxPool1D(Module):
    def __init__(self,k_size,stride):
        self._k_size = k_size
        self._stride = stride
        self._max = None

    def forward(self, X):
        size = X.shape
        res = np.zeros((size[0],(size[1] - self._k_size) // self._stride + 1,size[2]))
        # print(X.shape,res.shape)
        size = res.shape
        self._max = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                for l in range(size[2]):
                    stride = j*self._stride
                    # print(i,j,l,X[i][stride:stride+self._k_size],X.shape,stride)
                    res[i,j,l] = np.max([np.max(X[i][n][l]) for n in range(stride,stride+self._k_size)])
                    self._max[i,j,l] = int(np.argmax(X[i][n][l] for n in range(stride,stride+self._k_size) ) + stride)
        return res

    def backward_delta(self, input, delta):
        res = np.zeros(input.shape)
        size = delta.shape
        for i in range(size[0]):
            for j in range(size[1]):
                for l in range(size[2]):
                    res[i][int(self._max[i,j,l])][l] = delta[i,j,l]
        return res

    def backward_update_gradient(self, input, delta):
        pass
    def update_parameters(self, gradient_step=1e-3):
        pass

class Flatten(Module):
    def init(self):
        pass

    def forward(self, X):
        x, y, z = X.shape
        return X.reshape(x, y - z)

    def update_parameters(self, gradient_step=None):
        pass

    def backward_update_gradient(self, input=None, delta=None):
        pass

    def backward_delta(self, input, delta):
        return delta.reshape(input.shape)
