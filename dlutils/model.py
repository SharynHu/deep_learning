import torch
from torch import nn

class RNN(nn.Module):
    # def __init__(self, num_hiddens, vocab_size, init_state):
    #     super().__init__()
    #     self.hlayer = nn.
        
    # def forward()
    pass



# Softmax Regression
def mysoftmax(X):
    # softmax for X in shape [batch_size, d]
    X_exp = torch.exp(X)
    X_sum = X_exp.sum(1, keepdim=True)
    return X_exp/X_sum

class SoftmaxRegression(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, X):
        X_flat = self.flatten(X)
        return mysoftmax(self.linear(X_flat))
    
    def reset_weights(self):
        for name, parameter in self.named_parameters():
            torch.nn.init.normal_(parameter)



# Dropout layer for zeroing out inputs
class Dropout(nn.Module):
    def __init__(self, p, training = False) -> None:
        assert(0<=p<=1)
        super().__init__()
        self.p_zeroed = p
        self.train = training
    
    def forward(self, X):
        if self.p_zeroed==0:
            return X
        if self.p_zeroed==1:
            return torch.zeros_like(X)
        if self.training:
            # generating the mask, in this case the mask would be 
            # different for each sample in a minibatch
            p_mask = torch.ones_like(X)*self.p_zeroed
            X = torch.bernoulli(p_mask)*X
            return X/(1-self.p_zeroed)
        else:
            return X
    


class Conv2D(nn.Module):
    def __init__(self, kernel_size=(3,3)) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        # all nuerons share the same kernel and bias
        self.bias = nn.Parameter(torch.rand(1))

    def forward(self, X):
        """
        Here x is of shape [batch_size, n_h, n_w]
        """
        k_h, k_w = self.weight.shape
        N, n_h, n_w = X.shape
        Y = torch.empty(size=(N, n_h-k_h+1, n_w-k_w+1))
        for i in range(N):
            for j in range(n_h-k_h+1):
                for k in range(n_w-k_w+1):
                    Y[i][j][k] =  (X[i, j:j+k_h, k:k+k_w]*self.weight).sum()
        return Y


class Conv2DOnebyOne(nn.Module):
    """
    One by one 2d convolution for multi-channeled input and multi-channeled output.
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        # The weight matrix is of size
        self.linear = nn.Linear(c_in, c_out)
        self.c_in = c_in
        self.c_out = c_out

    def forward(self, X):
        c_in, n_h, n_w = X.shape
        X = torch.reshape(X, shape=(-1, n_h*n_w))
        X = torch.transpose(X, 0, 1)
        Y = self.linear(X)
        Y = torch.transpose(Y, 0, 1)
        Y = torch.reshape(Y, shape=(self.c_out, n_h, n_w))
        return Y

