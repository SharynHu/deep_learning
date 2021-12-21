import torch


def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()

def train_3ch(net, loss, num_epochs, train_loader, optimizer=None, test_loader=None, device=None):
    if device:
        net = net.to(device)
        loss = loss.to(device)
    if not optimizer:
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0)
    net.train()
    for i in range(num_epochs):
        for X,y in train_loader:
            if device:X, y = X.to(device)                                                                                                                                                                                                                                                                                                               , y.to(device)
            y_hat = net(X)
            ce_loss = loss(y_hat, y)
            optimizer.zero_grad()
            ce_loss.backward()
            optimizer.step()
    
         #after each epoch, calculate traning loss, testing loss and accuracy
        with torch.no_grad():
            net.eval()
            train_loss = 0
            test_loss = 0
            train_cp = 0
            test_cp = 0
            for X, y in train_loader:
                if device:X,y = X.to(device), y.to(device)
                y_hat = net(X)
                train_loss += loss(y_hat,y)
                train_cp += classifier_accuracy(y_hat, y)
            train_loss = train_loss/len(train_loader.dataset)
            train_cp = train_cp/len(train_loader.dataset)
            output_str = "epoch {i}, training loss {train_loss:f}, training accuracy {train_cp:f}".format(i=i,train_loss=train_loss, train_cp=train_cp)

            if test_loader:
                for X, y in test_loader:
                    if device:X,y = X.to(device), y.to(device)
                    y_hat = net(X)
                    test_loss += loss(y_hat,y)
                    test_cp += classifier_accuracy(y_hat, y)
                test_loss = test_loss/len(test_loader.dataset)
                test_cp = test_cp/len(test_loader.dataset)
                output_str += ", testing loss {test_loss:f}, testing accuracy {test_cp:f}".format(test_loss=test_loss, test_cp=test_cp)
            print(output_str)




def classifier_accuracy(y_hat, y):
    y_hat = y_hat.argmax(axis=1)
    cmp  = (y_hat.type(y.dtype)==y)
    return cmp.type(y.dtype).sum()

