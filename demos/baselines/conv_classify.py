import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np

from cosmoNODE.loaders import Anode as A
'''
The following program is a 1D convolutional neural network
that defines the kernel_size recursively.
Currently, I'm feeding in data that is 2D but I flatten it.
This is problematic and needs to be fixed.

# TODO:
    - port to 2DConv
    - test on MNIST
'''

class Conv1DNet(nn.Module):
    def __init__(self, input_dim=704, output_dim=14):
        super(Conv1DNet, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim

        self.delta = input_dim - output_dim
        self.ksizes = get_ksizes(self.delta)

        self.layers = []
        self.dims = []
        self.x = torch.ones([1, 1, self.in_dim])
        self.get_layers()

        self.real_layer_count = len(self.layers)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == self.real_layer_count - 3:
                break
            x = F.relu(layer(x))

        x = x.view(1,-1)
        x = self.model[-3](x)
        x = self.model[-2](x)
        x = self.model[-1](x)
        # x = self.model[i](x.flatten())
        # x = self.model[-1](x)  # softmax
        return x.double()

    def get_layers(self):
        prev = self.x

        prev_channels = 1
        channels = 1

        for i, ksize in enumerate(self.ksizes):
            if channels < 64:
                channels = prev_channels * 2

            layer = nn.Conv1d(prev_channels, channels, kernel_size=ksize)

            prev_channels = channels

            prev = layer(prev)

            self.dims.append(prev.shape)
            self.layers.append(layer)

        print(self.dims)
        conv_out = self.dims[-1][-1] * channels
        pool_size = math.floor(math.log2(conv_out))
        pool_layer = nn.MaxPool1d(pool_size)
        pool_out = pool_layer(prev).shape[-1]

        self.layers.append(pool_layer)
        self.layers.append(nn.Linear(pool_out * channels, self.out_dim))
        self.layers.append(nn.Softmax(dim=-1))


class Conv2DNet(nn.Module):
    def __init__(self, input_shape=(28, 28), out_classes=10):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)  # 24 -> 12
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)  # 8 -> 4
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def get_ksizes(delta):
    ksizes = []
    rough_layer_count = math.log2(delta)
    num_layers = round(rough_layer_count)
    for i in range(num_layers):
        pow = num_layers - i
        ksize = (2 ** (pow - 1)) + 1
        print(ksize)
        ksizes.append(ksize)
    return ksizes


if __name__ == '__main__':
    epochs = 1
    loader = A()

    x, y = loader.__getitem__(0)
    flat_x = x.flatten()
    net = Conv1DNet(flat_x.shape[0], y.shape[0]).double()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    logging_rate = 50

    for i in range(1, epochs + 1):
        for j, (x, y) in enumerate(loader):
            optimizer.zero_grad()

            with torch.no_grad():
                flat_x = x.flatten()
                reshaped_x = flat_x.reshape([1, 1, -1])

            pred = net(x)

            loss = criterion(y, pred)

            loss.backward()
            optimizer.step()
            if j % 50 == 0:
                with torch.no_grad():
                    print(f'pred: {pred} \n y: {y} \n loss: {loss} \n')

    torch.save(net.state_dict(), './demos/baselines/saved_models/conv_classify.pt')
