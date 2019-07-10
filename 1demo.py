import os
import argparse
import time
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import loaders as l

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--method', type=str, choices=['dopri5', 'adams'], default='dopri5')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--test_freq', type=int, default=5)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

'''
This file reads 'cosmoNODE/data/single_obj.csv' into a pytorch DataLoader
and attempts to use an ODE solver to learn the vector field.
'''

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.net = self.net.double()

        '''
        TODO: Look into what the nn.init.normal_ and nn.init.constant_ do.
        '''
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)  # y**3 is element cubing, no change of shape


def fl():
    flux_loader = l.FluxLoader()
    return flux_loader


def flux_item(index=20):
    flux_loader = fl()
    item = flux_loader.__getitem__(index)
    t = item[0]
    y = item[1]  # assumes 1D y data
    # for N-d y, make df.drop(t_column)
    return t, y


if __name__ == '__main__':



    ii = 0

    flux_loader = fl()

    t, y = flux_item()

    seq_len = len(t)

    true_t0 = t[0].reshape([1])
    true_y0 = y[0].reshape([1])

    func = ODEFunc().double()

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    niters = 1
    batch_time = 340
    test_freq = 1
    epochs = 100

    plt_x = []
    plt_y = []  # must be 1D
    real_y = []

    plt_loss_x = []
    plt_loss_y = []
    for epoch in range(1, epochs + 1):
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()

            train_pos = epoch * itr
            up_bound = itr*args.batch_time

            if up_bound >= seq_len:
                break

            batch_t = t[up_bound - args.batch_time:up_bound]

            pred_y = odeint(func, true_y0, batch_t)  # going to return batch_time # of predictions

            batch_y = y[up_bound - args.batch_time:up_bound]

            # list appends for graphing
            real_y += batch_y.tolist()
            plt_y += pred_y.tolist()
            plt_x += batch_t.tolist()

            loss = torch.mean(torch.abs(pred_y - batch_y)).requires_grad_(True)
            loss.backward()
            optimizer.step()

            if itr % args.test_freq == 0:
                with torch.no_grad():
                    pred_y = odeint(func, true_y0, batch_t)
                    loss = torch.mean(torch.abs(pred_y - batch_y))
                    print('Iter {:04d} | Epoch {} | Train pos {}| Total Loss {:.6f}'.format(itr, epoch, train_pos, loss.item()))
                    plt_loss_x.append(train_pos)
                    plt_loss_y.append(loss.item())
                    ii += 1

            end = time.time()

    # write testing loop

    print('training over')

    test_itrs = 1000
    min_time = flux_loader.df['mjd'].min()
    max_time = flux_loader.df['mjd'].max()
    print(min_time, max_time)
    test_times = torch.linspace(min_time, max_time, test_itrs)

    test_plt_y = []
    test_plt_x = []

    # for test_itr in range(1, test_itrs + 1):
    with torch.no_grad():

      up_bound = test_itrs - 1

      batch_t = test_times[0:up_bound]

      pred_y = odeint(func, true_y0, batch_t)  # going to return batch_time # of predictions

      test_plt_y += pred_y.tolist()
      test_plt_x += batch_t.tolist()
