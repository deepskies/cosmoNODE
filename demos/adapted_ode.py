import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cosmoNODE.loaders import LC


device = torch.device('cpu')

''' this is an adapted version of ricky's ode_demo.py '''

adjoint = True

if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

viz = True
viz_at_end = True
data_size = 350
batch_time = 25
batch_size = 175

test_frac = 0.2

test_freq = 5

lc = LC()

def ode_batch(time_sols, flux_sols):
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=True))
#     print(s)
    batch_y0 = flux_sols[s]  # (M, D)
    batch_t = time_sols[:batch_time]  # (T)
    batch_y = torch.stack([flux_sols[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

# todo
class Runner:
    def __init__(self):
        pass

num_curves = len(lc)

curve = lc[np.random.choice(num_curves)][['mjd', 'flux']]
times = torch.tensor(curve['mjd'].values.tolist())
flux_list = torch.tensor(curve['flux'].values.tolist())

t_0 = times[0]  # odeint takes time as 1d
flux_0 = flux_list[0].reshape(1, -1) # torch.Size([1, 1])

fluxes = flux_list.reshape(-1, 1, 1)

data_size = len(fluxes)  # - 1

batch_time = data_size//10
batch_size = data_size

def split_rand(length, test_frac=test_frac):
    # given int (representing timeseries length) and fraction to sample
    # returns np array of ints corresponding to the indices of the data
    # i'm not passing the data itself to this because i imagine that it would be slower

    test_indices = np.random.shuffle(indices)[:split_index]
    train_indices = np.delete(indices, test_indices)
    return train_indices, test_indices

def split_cutoff(length, test_frac=test_frac):
    indices, split_idx = split_index(length, test_frac)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    return train_indices, test_indices


def split_index(length, test_frac):
    indices = np.arange(length)
    split_idx = round(test_frac * length)
    return indices, split_idx
