import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cosmoNODE import utils
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

test_frac = 0.5
split_type = 'rand'
test_freq = 5

lc = LC(groupby_cols=['object_id'])

def ode_batch(time_sols, flux_sols):
    s = torch.from_numpy(np.random.choice(np.arange(train_size - batch_time, dtype=np.int64), batch_size, replace=True))
#     print(s)
    batch_y0 = flux_sols[s]  # (M, D)
    batch_t = time_sols[:batch_time]  # (T)
    batch_y = torch.stack([flux_sols[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def viz(pred_interpolation):
    plt.cla()
    plt.ylim(fluxes.min(), fluxes.max())
    plt.xlim(times.min(), times.max())
    plt.scatter(train_times.numpy(), train_fluxes.numpy(), c='b')
    plt.scatter(test_times.numpy(), test_fluxes.numpy(), c='r')
    plt.plot(eval_times.tolist(), pred_interpolation.flatten().tolist())
    plt.draw()
    plt.pause(1e-3)

# todo
class Runner:
    def __init__(self):
        pass

class ODEFunc(nn.Module):

    def __init__(self, dim):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(dim, 50),
            nn.Tanh(),
            nn.Linear(50, dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)

num_curves = len(lc)

curve = lc[np.random.choice(num_curves)][['mjd', 'flux']]
# curve = lc[0][['mjd', 'flux']]
times = torch.tensor(curve['mjd'].values.tolist())
flux_list = torch.tensor(curve['flux'].values.tolist())

t_0 = times[0]  # odeint takes time as 1d
flux_0 = flux_list[0].reshape(1, -1) # torch.Size([1, 1])

data_size = len(times) - 1
split_idx = round(test_frac * data_size)


if split_type == 'cutoff':
    train, test = utils.split_cutoff(data_size, test_frac)

    train_times = torch.tensor([times[train_elt] for train_elt in train])
    test_times = torch.tensor([times[test_elt] for test_elt in test])

    train_fluxes = torch.tensor([flux_list[train_elt] for train_elt in train])
    test_fluxes = torch.tensor([flux_list[test_elt] for test_elt in test])

    train_fluxes_shaped = train_fluxes.reshape(-1, 1, 1)
    test_fluxes_shaped = test_fluxes.reshape(-1, 1, 1)


if split_type == 'rand':
    train, test = utils.split_rand(data_size, test_frac)

    train_times = torch.tensor([times[train_elt] for train_elt in train])
    test_times = torch.tensor([times[test_elt] for test_elt in test])

    train_fluxes = torch.tensor([flux_list[train_elt] for train_elt in train])
    test_fluxes = torch.tensor([flux_list[test_elt] for test_elt in test])

    train_fluxes_shaped = train_fluxes.reshape(-1, 1, 1)
    test_fluxes_shaped = test_fluxes.reshape(-1, 1, 1)

fluxes = flux_list.reshape(-1, 1, 1)

train_size = len(train_times)
print(f'train_size: {train_size}')
batch_time = train_size // 4
batch_size = train_size // 2

print(f'train_times: {train_times}')
print(f'train_fluxes: {train_fluxes}')

epochs = 5
niters = 100
dim = lc.dim
odefunc = ODEFunc(dim)
optimizer = optim.RMSprop(odefunc.parameters(), lr=1e-2)
ii = 0
losses = []

# used for plotting
eval_times = torch.linspace(times.min(), times.max(), data_size*20)

r_tol = 1e-1
a_tol = 1e-1

by0_f, bt_f, by_f = ode_batch(train_times, train_fluxes_shaped)
print(f'by0.shape : {by0_f.shape}, bt.shape: {bt_f.shape}, by.shape: {by_f.shape}')

plt.ion()

for epoch in range(1, epochs + 1):
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        by0_f, bt_f, by_f = ode_batch(train_times, train_fluxes_shaped)
        pred_f = odeint(odefunc, by0_f, bt_f, rtol=r_tol, atol=a_tol)
        loss = torch.mean(torch.abs(pred_f - by_f))
        loss.backward()
        optimizer.step()
        if itr % test_freq == 0:
            with torch.no_grad():
                pred_interpolation = odeint(odefunc, flux_0, eval_times, rtol=r_tol, atol=a_tol)
                pred_f = odeint(odefunc, flux_0, times, rtol=r_tol, atol=a_tol)
                loss = torch.mean(torch.abs(pred_f - fluxes))
                losses.append(loss)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if viz:
                    viz(pred_interpolation)
                ii += 1

if viz_at_end:
    print(losses)
    loss_over_time = [i for i in range(len(losses))]
    plt.plot(loss_over_time, losses)
    plt.show()
    plt.pause(5)
    plt.ioff()
