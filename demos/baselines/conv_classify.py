import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

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
    def __init__(self, input_dim=704, output_dim=14, batching_size=12):
        super(Conv1DNet, self).__init__()
        self.in_dim = input_dim
        self.out_dim = output_dim
        self.batching_size = batching_size

        self.delta = input_dim - output_dim
        self.ksizes = get_ksizes(self.delta)

        self.layers = []
        self.dims = []
        self.x = torch.ones([self.batching_size, 1, self.in_dim])
        self.get_layers()

        self.real_layer_count = len(self.layers)
        self.model = nn.ModuleList(self.layers)

    def forward(self, x):
        for i, layer in enumerate(self.model):
            if i == self.real_layer_count - 3:
                break
            x = F.relu(layer(x))
            # print(x.shape)

        x = x.view(self.batching_size, 1, -1)  # flatten to 1d before pooling
        # print(x.shape)
        x = self.model[-3](x)  # pool_layer

        x = x.view(self.batching_size, -1)

        x = self.model[-2](x)  # linear
        # print(x.shape)
        x = self.model[-1](x)  # softmax

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

        numel_wo_batch = self.dims[-1][1] * self.dims[-1][2]
        print(numel_wo_batch)
        # conv_out = self.dims[-1].numel()
        pool_ksize = math.floor(math.log2(numel_wo_batch))

        pool_layer = nn.MaxPool1d(pool_ksize)

        pool_out = pool_layer(prev.view(self.batching_size, 1, -1)).shape

        linear_input_dim = pool_out[-1]

        self.layers.append(pool_layer)
        self.layers.append(nn.Linear(linear_input_dim, self.out_dim))
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
    epochs = 10
    dataset = A()
    x, y = dataset.__getitem__(0)
    flat_x = x.flatten()
    print(f'input_dim: {flat_x.shape[0]}')

    batching_size = math.floor(math.log2(len(dataset)))

    validation_split = .5
    shuffle_dataset = True
    random_seed= 42

    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    data_loader = DataLoader(dataset, batch_size=batching_size, sampler=train_sampler)

    test_loader = DataLoader(dataset, batch_size=batching_size, sampler=valid_sampler)

    net = Conv1DNet(flat_x.shape[0], y.shape[0], batching_size).double()
    net.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    logging_rate = 10
    losses = []

    for i in range(1, epochs + 1):
        for j, (x, y) in enumerate(data_loader):
            optimizer.zero_grad()

            pred = net(x)

            loss = criterion(pred, torch.max(y.long(), 1)[1])

            loss.backward()
            optimizer.step()
            if j % 50 == 0:
                with torch.no_grad():
                    losses.append(loss)
                    # print(f'pred: {pred} \n y: {y} \n loss: {loss} \n')

    for loss_val in losses:
        print(f'loss_val {loss_val}\n')

    # with torch.no_grad():
    #     for h, (x, y) in enumerate(test_loader):

    # print(losses)
    # torch.save(net.state_dict(), './demos/baselines/saved_models/conv_classify.pt')
