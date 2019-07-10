import os
import argparse
import time

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import loaders as l
import macros as m

from torchdiffeq import odeint_adjoint as odeint

'''
Written explanation before writing code.
1D_demo.py requires that the y data that is being solved for is 1 dimensional,
hence the name.

Now we want to try to learn N-Dimensional time series, eg. including other features
from the LSST Kaggle set (not just flux).

My hope is that we can then try to train one huge ODE on the whole training set
and see what happens. I imagine that we will begin to see the limits of NODEs.



'''

class ODEFunc(nn.Module):
    '''
    What is the best way to take in the dimension of y and have the model adapt
    the input shape?
    Should it just be rank?

    This is unresolved, but I think that I am just going to have y_dimension
    be the # of features.
    However, this is probably not the best way as we will want to batch etc.
    '''

    def __init__(self, y_dimension):
        super(ODEFunc, self).__init__()
        '''
        Layer defns here are arbitrary, ideally make them a learned value.
        A simple improvement could be simply making them a function of the
        input size. Eg. y_dimension * 10 (although the 10 is arbitrary)
        '''

        self.net = nn.Sequential(
            nn.Linear(y_dimension, 50),
            nn.Tanh(),
            nn.Linear(50, y_dimension),
        )

        self.net = self.net.double()

    def forward(self, t, y):
        return self.net(y**3)


'''
Next, we need to load in the data from the training set.

Initialize a DataLoader to iterate over during training.

todo: test df.df w MinMaxScaler vs no scaling

'''

'''
df schema:
['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected', 'ra', 'decl',
'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz',
'hostgal_photoz_err', 'distmod', 'mwebv', 'target'],

since target is categorical data, it wouldn't make sense to include this
in the ODE.

Aside from actually figuring out what data to give it, the loader runs with
batch size of 1, but now the infrastructure for actual training needs to be written.


for now ND with single object
'''


if __name__ == '__main__':
    BATCH_SIZE = 16

    data_loader = l.NDim(BATCH_SIZE)

    # obj = data_loader.raw[data_loader.raw['object_id'] == 615]

    # t_df = obj['mjd']
    # y_df = obj.drop(['object_id', 'mjd', 'target'], axis=1)
    #
    # t = torch.tensor(t_df.values)
    # y = torch.tensor(y_df.values)



    # y_dim = len(data_loader.y.columns)
    # print(data_loader.df.columns)
    # print(data_loader.y.columns)

    func = ODEFunc(data_loader.y_dim).double()

    print(func)

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)

    for i, item in enumerate(data_loader):
        t = item[0]
        y = item[1]

        t0 = t[0].reshape([1])
        y0 = y[0].reshape([-1])

        if t is None:
            break

        optimizer.zero_grad()

        batch_t = t[0:BATCH_SIZE]
        print(batch_t.shape)

        batch_y = y[0:BATCH_SIZE]
        print(batch_t.shape)

        pred_y = odeint(func, y0, batch_t)

        loss = torch.mean(torch.abs(pred_y - batch_y)).requires_grad_(True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            print(f'loss: {loss}')
            # print(f'iter: {0}, t: {batch_t}, y: {batch_y}')
