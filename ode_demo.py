import os
import argparse
import time
import numpy as np

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
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
args = parser.parse_args()

if args.adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


# t = torch.linspace(0., 25., args.data_size) # objects times 


true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])  # ???


flux_loader = l.FluxLoader()
item = flux_loader.__getitem__(20)
t = item[0]

true_y = item[1]
true_y0 = true_y[0]

def get_batch(itr):
    y_cutoff = args.batch_size * itr
    y_lower_cutoff = y_cutoff - args.batch_size

    s = torch.tensor([i for i in range(y_lower_cutoff, y_cutoff)])

    batch_y0 = true_y[y_lower_cutoff:y_cutoff]  # read from dataframe
    
    t_cutoff = args.batch_time * itr
    t_lower_cutoff = t_cutoff - args.batch_time

    batch_t = t[t_lower_cutoff:t_cutoff]  # (T)
    
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


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

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)


if __name__ == '__main__':

    ii = 0

    func = ODEFunc()
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    

    for itr in range(1, args.niters + 1):
        
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch(itr)

        pred_y = odeint(func, batch_y0, batch_t)
        
        loss = torch.mean(torch.abs(pred_y - batch_y))
        
        loss.backward()
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, t)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y, pred_y, func, ii)
                ii += 1

        end = time.time()