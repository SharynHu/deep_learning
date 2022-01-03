import torch
from torch.nn import Module
from torch.utils.data import DataLoader
import torchvision
import sys
sys.path.append("../dlutils")
import importlib
import model
import loss
import train
import dataset
importlib.reload(model)
importlib.reload(loss)
importlib.reload(train)
importlib.reload(dataset)
    
# get Fashion-MNIST Dataset
from dataset import load_fashion_mnist_dataset
batch_size = 256
train_loader,test_loader=load_fashion_mnist_dataset(batch_size)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt',
        'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

# define the model
from model import SoftmaxRegression
num_train_samples, height, width = list(train_loader.dataset[0][0].shape)
in_features = height*width
out_features = 10
net = SoftmaxRegression(in_features, out_features)


# define the loss
from loss import CrossEntropyLoss
loss = CrossEntropyLoss()


from train import sgd, classifier_accuracy
num_epochs = 10
lr = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.reset_weights()
net.to(device)
loss.to(device)


# training
for i in range(num_epochs):
    for X,y in train_loader:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        ce_loss = loss(y_hat, y).sum()
        ce_loss.backward()
        sgd(net.parameters(), lr, batch_size)
    
    #after each epoch, calculate traning loss, testing loss and accuracy
    with torch.no_grad():
        train_loss = 0
        test_loss = 0
        train_cp = 0
        test_cp = 0
        for X, y in train_loader:
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            train_loss += loss(y_hat,y).sum()
            train_cp += classifier_accuracy(y_hat, y)
        train_loss = train_loss/len(train_loader.dataset)
        train_cp = train_cp/len(train_loader.dataset)
            
        for X, y in test_loader:
            X,y = X.to(device), y.to(device)
            y_hat = net(X)
            test_loss += loss(y_hat,y).sum()
            test_cp += classifier_accuracy(y_hat,y)
        test_loss = test_loss/len(test_loader.dataset)
        test_cp = test_cp/len(test_loader.dataset)
        print(f'epoch {i}, training loss {float(train_loss):f}, training accuracy {float(train_cp):f},testing loss {float(test_loss):f}, testing accuracy {float(test_cp):f}')
