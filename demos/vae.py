

import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import h

'''
This network is defined recursively.
|layers| ~ log_2(input_dim)
'''
class Net(nn.Module):
    def __init__(self, input_dim):
        super(Net, self).__init__()
        self.input_dim = input_dim

        self.encoder = []
        self.decoder = []

        self.instantiate_network()

        self.enc = nn.ModuleList(self.encoder)
        self.dec = nn.ModuleList(self.decoder)

    def instantiate_network(self):

        prev = self.input_dim
        cur = self.input_dim

        tuples = []

        while cur != 1:
            cur = prev // 2
            tuples.append((prev, cur))
            prev = cur

        print(tuples)

        for tup in tuples:
            self.encoder.append(nn.Linear(tup[0], tup[1]))

        for tup in tuples[::-1]:
            self.decoder.append(nn.Linear(tup[1], tup[0]))


    def forward(self, x, direction):
        for layer in direction:
            x = F.relu(layer(x))
        return x

    def __repr__(self):
        print(f'encoder: {self.enc}')
        print(f'decoder: {self.dec}')
        return 'network'


if __name__ == '__main__':

    data_loader = Loader()
    net = Net(data_loader.length).double()

    print(net)
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
    epochs = 1

    net.train()

    for i in range(1, epochs + 1):
        for j, (x, _) in enumerate(data_loader):
            optimizer.zero_grad()

            encoded = net.forward(x, net.enc)
            decoded = net.forward(encoded, net.dec)

            loss = torch.abs(x - decoded)

            loss.backward()
            optimizer.step()
