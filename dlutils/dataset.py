import math
import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
import os
import collections
import re
import urllib.request
import random

# The FashionMNIST dataset
def load_fashion_mnist_dataset(batch_size, resize=None, home="/scratch/home/acct-hpc/hpchxj"):
    trans = [torchvision.transforms.ToTensor()]
    if resize:
        trans.insert(0, torchvision.transforms.Resize(resize))
    trans = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.FashionMNIST(root=home+"/data", train=True, transform=trans, download=True)
    mnist_test = torchvision.datasets.FashionMNIST(root=home+"/data", train=False, transform=trans, download=True)
    mnist_train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=4)
    mnist_test_loader = DataLoader(mnist_test, batch_size=batch_size,shuffle=True, num_workers=4)
    return mnist_train_loader,mnist_test_loader


def tokenize(lines, token):
    if token == 'word':
        return [line.split() for line in lines]
    elif token == 'char':
        return [list(line) for line in lines]
    else:
        print('ERROR: unknown token type: ' + token)
        

class Vocab(object):
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter =collections.Counter([token for line in tokens for token in line])
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {
            token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1
    
    def __len__(self):
        return len(self.idx_to_token)
    
    def   __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx[tokens]
        return [self.token_to_idx[token] for token in tokens]
    
    def to_tokens(self, dices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]
    
    
                 
class TimeMachine(torch.utils.data.Dataset):
    def __init__(self, num_steps, path="./", token="word"):
        URL = "https://www.gutenberg.org/files/35/35-0.txt"
        if not os.path.exists(path):
            os.makedirs("data")
        urllib.request.urlretrieve(URL, os.path.join(path, "the_time_machine.txt"))
        
        with open("data/the_time_machine.txt", "r") as f:
            lines = f.readlines()
        lines = [re.sub('[^A-Za-z]+', ' ', line).strip().lower() for line in lines]  
        
        tokens = tokenize(lines, token)
        vocab = Vocab(tokens)
        corpus = [vocab[token] for line in tokens for token in line]
        
        # start with a random offset
        corpus = corpus[random.randint(0, num_steps-1):]
        self.corpus = corpus
        self.vocab = vocab
        self.num_steps = num_steps
        
    def __len__(self):
        num_subseqs = (len(self.corpus)-1)//self.num_steps
        return num_subseqs
    
    def __getitem__(self, index):
        return corpus[index*self.num_steps:(index+1)*self.num_steps], corpus[(index+1)*self.num_steps]
    
    
    
        
        
        
        
        

    