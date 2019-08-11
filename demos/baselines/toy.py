import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from cosmoNODE import utils

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(1, 1)
        self.l2 = nn.Linear(1, 1)
        self.l3 = nn.Linear(1, 1)


    def forward(self, x):
        out = F.relu(self.l1(x))
        out = F.relu(self.l2(out))
        out = F.relu(self.l3(out))
        return out

if __name__ == '__main__':
    epochs = 5
    niters = 1000
    test_freq = 500

    net = Net()
    optimizer = optim.RMSprop(net.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    losses = []

    np_t = np.linspace(1, 50, 1000)
    np_y = np.sin(np_t)

    t = torch.tensor(np_t, dtype=torch.float)
    y = torch.tensor(np_y, dtype=torch.float)

    train_size = len(t)
    batch_time = 10
    batch_size = 100


    batch_y0, batch_t, batch_y = utils.ode_batch(y, t, train_size, batch_time, batch_size)
    print(f'by0.shape : {batch_y0.shape}, bt.shape: {batch_t.shape}, by.shape: {batch_y.shape}')
    print(f'by0.dtype : {batch_y0.dtype}, bt.dtype: {batch_t.dtype}, by.dtype: {batch_y.dtype}')

    plt.ion()

    for epoch in range(1, epochs+1):
        for itr in range(1, niters+1):
            loc = (itr * epoch) % niters
            x = t[loc]
            real_y = y[loc]
            pred_y = net(x.view(-1,1))
            loss = criterion(pred_y, real_y)

            loss.backward()
            print(loss.item())

            if loc % test_freq == 0:
                all_pred_ys = []

                for j in t:
                    pred_y = net(j.view(-1,1))
                    all_pred_ys.append(pred_y)

                all_ys_tensor = torch.tensor(all_pred_ys)
                total_loss = criterion(all_ys_tensor, y)
                plt.scatter(t.numpy(), all_ys_tensor.numpy())
                plt.draw()
                print(total_loss)
        plt.ioff()
