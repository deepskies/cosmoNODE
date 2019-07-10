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


class NDim(Dataset):
    def __init__(self):
        fns = ['training_set', 'training_set_metadata']
        self.tr, self.tr_meta = m.read_multi(fns)

        self.raw = pd.merge(self.tr, self.tr_meta, on='object_id')
        self.raw = self.raw.fillna(0)

        # is it hacking to give the model obj_id?
        self.obj_ids = self.raw['object_id']

        self.df = self.raw.drop(['object_id', 'target'], axis=1)

        self.t = self.df['mjd']  # 1D list of values to calculate Y for in ODE
        self.y = self.df.drop('mjd', axis=1)

        self.train_len = len(self.df)

    '''
    What shape should __getitem__ return?
    Returning a single line seems inefficient. Fix later
    For now im batching in here
    '''
    def __getitem__(self, index):
        return (self.t.iloc[index], self.y.iloc[index])

    def __len__(self):
        return self.train_len


'''
df schema:
['object_id', 'mjd', 'passband', 'flux', 'flux_err', 'detected', 'ra', 'decl',
'gal_l', 'gal_b', 'ddf', 'hostgal_specz', 'hostgal_photoz',
'hostgal_photoz_err', 'distmod', 'mwebv', 'target'],

since target is categorical data, it wouldn't make sense to include this
in the ODE
'''


if __name__ == '__main__':
    BATCH_SIZE = 100

    data_loader = NDim()
    y_dim = len(data_loader.y.columns)
    print(data_loader.df.columns)
    print(data_loader.y.columns)
    data_generator = inf_generator(data_loader)

    func = ODEFunc(y_dim)
    print(func)

    # for i, (t, y) in enumerate(data_generator):
