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
    

        


        
    