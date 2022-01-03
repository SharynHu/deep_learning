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

batch_size=256
train_loader,test_loader = load_fashion_mnist_dataset(batch_size=batch_size)

from model import Dropout

p1_zeroed, p2_zeroed = 0.2, 0.5

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(), torch.nn.Linear(784, 256), torch.nn.ReLU(),
            # Add a dropout layer after the first fully connected layer
            Dropout(p1_zeroed),
            torch.nn.Linear(256, 256), torch.nn.ReLU(),
            # Add a dropout layer after the second fully connected layer
            Dropout(p2_zeroed), 
            torch.nn.Linear(256, 10))
        
    def forward(self, X):
        return self.net(X)
    
    def train(self):
        for subnet in self.modules():
            if isinstance(subnet, Dropout):
                subnet.train = True
            
    def eval(self):
        for subnet in self.modules():
            if isinstance(subnet, Dropout):
                subnet.train = False

net = Net()
net.train()


num_epochs = 10
lr = 0.1
loss = torch.nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from train import train_3ch
train_3ch(net, loss, num_epochs, train_loader, optimizer=trainer, test_loader=test_loader, device=device)
