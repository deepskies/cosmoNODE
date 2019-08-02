import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cosmoNODE import utils
from cosmoNODE.loaders import LC

device = torch.device('cuda')

''' this is an adapted version of ricky's ode_demo.py '''

adjoint = True

if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def ode_batch(ts, ys):
    s = torch.from_numpy(np.random.choice(np.arange(train_size - batch_time, dtype=np.int64), batch_size, replace=True))
#     print(s)
    batch_y0 = ys[s]  # (M, D)
    batch_t = ts[:batch_time]  # (T)
    batch_y = torch.stack([ys[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def visualize(pred_interpolation):
    plt.cla()
    plt.ylim(ys.min(), ys.max())
    plt.xlim(times.min(), times.max())
    plt.scatter(train_times.numpy(), train_ys.numpy(), c='b')
    plt.scatter(test_times.numpy(), test_ys.numpy(), c='r')
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

viz = True
viz_at_end = True

test_frac = 0.5
split_type = 'cutoff'
test_freq = 5

lc = LC(cols=['mjd', 'flux', 'passband'], groupby_cols=['object_id'])

if lc.dim == 2:
    graph_3d = True
elif lc.dim > 2:
    # todo, this is a jank bug catcher
    viz = False
    viz_at_end = False
else:
    graph_3d = True

if viz:
    plt.ion()
    fig = plt.figure()
    
    if graph_3d:
        ax = fig.add_subplot(111, projection='3d')
    else:
        ax = fig.add_subplot(111)

    plt.draw()
    plt.pause(1e-3)


num_curves = len(lc)
curve = lc[np.random.choice(num_curves)]

# odeint takes time as 1d
t = curve['mjd']
t_list = t.tolist()
times = torch.tensor(t.values)

# exclude time from the data
ys = torch.tensor(curve.drop(['mjd'], axis=1).values)
# ys_list = ys.tolist()
# .values, dtype=torch.double)
ys_reshaped = ys.reshape(-1, 1, lc.dim)
# ys_list = ys_reshaped.tolist()

ys_list = list(ys_reshaped)
# print(f'ys_list : {ys_list.dtype}, ys_reshaped: {ys_reshaped.dtype}')
# flux_list = torch.tensor(curve['flux'].values.tolist())

t_0 = times[0]  # odeint takes time as 1d
y_0 = ys_reshaped[0]

data_size = len(times) - 1
split_idx = round(test_frac * data_size)

if split_type == 'cutoff':
    train, test = utils.split_cutoff(data_size, test_frac)

    ttrain = torch.tensor(train)

    ttest = torch.tensor(test)

    train_times = times[ttrain]
    test_times = times[ttest]

    train_ys = ys_reshaped[ttrain]
    test_ys = ys_reshaped[ttest]

    # train_times = torch.tensor([t_list[train_elt] for train_elt in train])
    # test_times = torch.tensor([t_list[test_elt] for test_elt in test])
    #
    # train_ys = torch.tensor([ys_list[train_elt] for train_elt in train])
    # test_ys = torch.tensor([ys_list[test_elt] for test_elt in test])

    train_ys_shaped = train_ys.reshape(-1, 1, lc.dim)
    test_ys_shaped = test_ys.reshape(-1, 1, lc.dim)


if split_type == 'rand':
    train, test = utils.split_rand(data_size, test_frac)

    train_times = torch.tensor([t_list[train_elt] for train_elt in train])
    test_times = torch.tensor([t_list[test_elt] for test_elt in test])

    train_ys = torch.tensor([ys_list[train_elt] for train_elt in train])
    test_ys = torch.tensor([ys_list[test_elt] for test_elt in test])

    train_ys_shaped = train_ys.reshape(-1, 1, lc.dim)
    test_ys_shaped = test_ys.reshape(-1, 1, lc.dim)


train_size = len(train_times)
print(f'train_size: {train_size}')

batch_time = train_size // 4
batch_size = train_size // 2

# print(f'train_times: {train_times}')
# print(f'train_ys: {train_ys}')

epochs = 5
niters = 100
dim = lc.dim
odefunc = ODEFunc(dim).double()
optimizer = optim.RMSprop(odefunc.parameters(), lr=1e-2)
ii = 0
losses = []

# used for plotting
eval_times = torch.linspace(times.min(), times.max(), data_size*20).double()
print(f'eval_times: {eval_times.dtype}')

r_tol = 1e-1
a_tol = 1e-1

print(f'train_times : {train_times.dtype}, train_ys_shaped: {train_ys_shaped.dtype}')
print(f'train_times.shape : {train_times.shape}, train_ys_shaped.shape: {train_ys_shaped.shape}')

by0_f, bt_f, by_f = ode_batch(train_times, train_ys_shaped)
print(f'by0.shape : {by0_f.shape}, bt.shape: {bt_f.shape}, by.shape: {by_f.shape}')
print(f'by0.dtype : {by0_f.dtype}, bt.dtype: {bt_f.dtype}, by.dtype: {by_f.dtype}')


for epoch in range(1, epochs + 1):
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        by0_f, bt_f, by_f = ode_batch(train_times, train_ys_shaped)
        # print(f'by0.shape : {by0_f.dtype}, bt.shape: {bt_f.dtype}, by.shape: {by_f.dtype}')
        pred_f = odeint(odefunc, by0_f, bt_f, rtol=r_tol, atol=a_tol)
        loss = torch.mean(torch.abs(pred_f - by_f))
        loss.backward()
        optimizer.step()
        if itr % test_freq == 0:
            with torch.no_grad():
                pred_interpolation = odeint(odefunc, y_0, eval_times, rtol=r_tol, atol=a_tol)
                pred_f = odeint(odefunc, y_0, times, rtol=r_tol, atol=a_tol)
                loss = torch.mean(torch.abs(pred_f - ys))
                losses.append(loss)
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if viz:
                    visualize(pred_interpolation)
                ii += 1

if viz_at_end:
    print(losses)
    loss_over_time = [i for i in range(len(losses))]
    plt.plot(loss_over_time, losses)
    plt.show()
    plt.pause(5)
    plt.ioff()
