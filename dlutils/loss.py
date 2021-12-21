from torch import nn
import torch


class CrossEntropyLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self,y_hat, y):
        # Note that here y_hat is a 2-d matrix and y is a vector
        return -(torch.log(y_hat[range(len(y_hat)), y]))