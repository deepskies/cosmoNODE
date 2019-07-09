import os
import argparse
import time
import numpy as np

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import loaders as l

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

Using loaders.py, initialize a DataLoader
'''
