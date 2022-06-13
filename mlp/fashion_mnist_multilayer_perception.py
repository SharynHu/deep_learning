import torch
from torch import nn
import torchvision
import sys
sys.path.append("../dlutils")
import importlib
import model
import dataset
import train
import loss
importlib.reload(model)
importlib.reload(dataset)
importlib.reload(train)
importlib.reload(loss)
from dataset import load_fashion_mnist_dataset
from train import train_3ch

batch_size = 256
train_mnist_loader, test_mnist_loader = load_fashion_mnist_dataset(batch_size)


# An MLP with one single hidden layer
class MLP(nn.Module):
    def __init__(self, in_features, out_features, num_hiddens):
        super().__init__()
        self.flatten = nn.Flatten()
        self.hidden = nn.Sequential(nn.Linear(in_features, num_hiddens), nn.ReLU())
        self.out = nn.Linear(num_hiddens, out_features)
    
    def forward(self, x):
        return self.out(self.hidden(self.flatten(x)))
    
in_features = 28*28
out_features = 10
num_hiddens = 128
mlp = MLP(in_features, out_features, num_hiddens)

optimizer = torch.optim.Adam(mlp.parameters(),lr=0.1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 10
train_3ch(mlp, loss, num_epochs,train_loader=train_mnist_loader, optimizer=optimizer,\
         test_loader=test_mnist_loader, device=device)