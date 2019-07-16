import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy

from cosmoNODE.loaders import Anode as A

class Net(nn.Module):
    def __init__(self, input_dim=704, output_dim=14):
        super(Net, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim

        self.delta = input_dim - output_dim
        self.rough_layer_count = math.log2(self.delta)
        self.num_layers = round(self.rough_layer_count)

        self.ksizes = []
        self.get_ksizes()

        self.layers = []
        self.dims = []
        self.x = torch.ones([1, 1, self.in_dim])
        self.get_layers()

        self.real_layer_count = len(self.layers)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == self.real_layer_count - 2:
                break
            x = F.relu(layer(x))
        x = self.model[i](x.flatten())
        x = self.model[-1](x)  # softmax
        return x.double()

    def get_ksizes(self):
        for i in range(self.num_layers):
            pow = self.num_layers - i
            ksize = (2 ** (pow - 1)) + 1
            print(ksize)
            self.ksizes.append(ksize)

    def get_layers(self):
        prev = self.x
        for ksize in self.ksizes:
            layer = nn.Conv1d(1, 1, kernel_size=ksize)
            prev = layer(prev)
            self.dims.append(prev.shape)
            self.layers.append(layer)
        print(self.dims)
        self.layers.append(nn.Linear(self.dims[-1][-1], self.out_dim))
        self.layers.append(nn.Softmax(dim=0))


if __name__ == '__main__':
    epochs = 1
    loader = A()

    x, y = loader.__getitem__(0)
    flat_x = x.flatten()
    net = Net(flat_x.shape[0], y.shape[0]).double()
    net.train()

    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)

    for i in range(1, epochs + 1):
        for j, (x, y) in enumerate(loader):
            optimizer.zero_grad()

            with torch.no_grad():
                flat_x = x.flatten()
                reshaped_x = flat_x.reshape([1, 1, -1])

            pred = net(reshaped_x)
            print(f'p: {pred}')
            print(f'y: {y}')

            loss = y - pred

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                print(f'pred: {pred} \n loss {loss}')
