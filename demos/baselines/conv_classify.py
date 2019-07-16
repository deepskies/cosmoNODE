import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy

# from cosmoNODE.loaders import Anode as A

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
        self.get_layers()
        self.real_layer_count = len(self.layers)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == self.real_layer_count - 1:
                break
            x = F.relu(layer(x))
        x = F.Softmax(self.model[i](x.Flatten())) 
        return x.double()

    def get_ksizes(self):
        for i in range(self.num_layers):
            pow = self.num_layers - i
            ksize = (2 ** (pow - 1)) + 1
            print(ksize)
            self.ksizes.append(ksize)

    def get_layers(self):
        for ksize in self.ksizes:
            self.layers.append(nn.Conv1d(1, 1, kernel_size=ksize))
        self.layers.append(nn.Linear(self.in_dim, self.out_dim))
# if __name__ == '__main__':
#     net = Net()
#     loader = A()
