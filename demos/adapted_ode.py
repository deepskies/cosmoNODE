import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from cosmoNODE import utils
from cosmoNODE.loaders import LC

device = torch.device('cpu')

''' this is an adapted version of ricky's ode_demo.py '''

adjoint = False

if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint



def ode_batch():
    s = torch.from_numpy(np.random.choice(np.arange(train_size - batch_time, dtype=np.int64), batch_size, replace=False))
#     print(s)
    batch_y0 = train_ys_shaped[s]  # (M, D)
    batch_t = train_times[:batch_time]  # (T)
    batch_y = torch.stack([train_ys_shaped[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def visualize(pred_interpolation):

    plt.cla()
    # if graph_3d:
    #     pass

    x = train_times.flatten().numpy()
    # even if we pass many columns to y we just take the first column (assumed to be flux)
    y = train_ys[:, 0].flatten().numpy()

    # for dim > 1, # todo
    # print(len(x))
    # print(len(y))
    # # y = train_ys[:, 0, 0].flatten().numpy()
    # print(len(x))
    # print(len(y))
    plt.scatter(x, y, c='b', s=0.5)
    plt.scatter(test_times.numpy(), test_ys[:, 0].numpy(), c='r', s=0.5)
    plt.plot(eval_times.tolist(), pred_interpolation[:, 0].flatten().tolist())

    plt.draw()
    plt.pause(1e-3)

class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, true_A)


# with torch.no_grad():
#     true_y = odeint(Lambda(), true_y0, t, method='dopri5')

# todo
class Runner:
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
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


lc = LC(cols=['mjd', 'flux', 'passband', 'flux_err', 'detected'], groupby_cols=['object_id'], meta=True)
# lc = LC(cols=['mjd', 'flux'])
dim = lc.dim
viz = True
viz_at_end = True

test_frac = 0.5
split_type = 'rand'
test_freq = 50

graph_3d = False

# if dim == 2:
#     graph_3d = True
# elif dim > 2:
#     # todo, this is a jank bug catcher
#     viz = False
#     viz_at_end = False
# else:
#     graph_3d = False

num_curves = len(lc)
curve = lc[0]
# curve = lc[np.random.choice(num_curves)]
curve = curve.sort_values(by='mjd')
dim = lc.dim
# x = np.linspace(0, 100, 1000)

# f = x * x

# f = 100 * np.cos(x/100)
# dim = 1
# odeint takes time as 1d
t = curve['mjd']
# t = x
t_list = t.tolist()
times = torch.tensor(t.values)
# times = torch.tensor(t).reshape(-1, 1)

# exclude time from the data
ys = torch.tensor(curve.drop(['mjd'], axis=1).values)

# ys = torch.tensor(f)
# ys_list = ys.tolist()
# .values, dtype=torch.double)

ys_reshaped = ys.reshape(-1, 1, dim)
# ys_list = ys_reshaped.tolist()

ys_list = list(ys_reshaped)
# print(f'ys_list : {ys_list.dtype}, ys_reshaped: {ys_reshaped.dtype}')
# flux_list = torch.tensor(curve['flux'].values.tolist())

t_0 = times[0]  # odeint takes time as 1d
y_0 = ys_reshaped[0]

data_size = len(times) - 1
split_idx = round(test_frac * data_size)

if viz:
    plt.ion()
    # fig = plt.figure()

    # if graph_3d:
    #     ax = fig.add_subplot(111, projection='3d')
    # else:
    #     ax = fig.add_subplot(111)

    # ax.ylim(ys.min(), ys.max())
    # ax.xlim(times.min(), times.max())
    plt.ylim(ys.min(), ys.max())
    plt.xlim(times.min(), times.max())
    plt.draw()
    plt.pause(1e-3)


if split_type == 'cutoff':
    train, test = utils.split(data_size, test_frac, type=split_type)

    train_times = times[train]
    test_times = times[test]

    train_ys = ys[train]
    test_ys = ys[test]

    train_ys_shaped = train_ys.reshape(-1, 1, dim)
    test_ys_shaped = test_ys.reshape(-1, 1, dim)


if split_type == 'rand':
    train, test = utils.split(data_size, test_frac, type=split_type)

    train_times = times[train]
    test_times = times[test]

    train_ys = ys[train]
    test_ys = ys[test]

    train_ys_shaped = train_ys.reshape(-1, 1, dim)
    test_ys_shaped = test_ys.reshape(-1, 1, dim)


train_size = len(train_times)
print(f'train_size: {train_size}')

batch_time = 10 #  train_size // 4
batch_size = train_size // 2 #  train_size // 2

epochs = 5
niters = 1000

odefunc = ODEFunc(dim).double()
optimizer = optim.RMSprop(odefunc.parameters(), lr=1e-3)
criterion = nn.MSELoss()
ii = 0
losses = []

# used for plotting
eval_times = torch.linspace(times.min(), times.max(), data_size*20).double()
print(f'eval_times: {eval_times.dtype}')

# r_tol = 1e-3
# a_tol = 1e-4

print(f'train_times : {train_times.dtype}, train_ys_shaped: {train_ys_shaped.dtype}')
print(f'train_times.shape : {train_times.shape}, train_ys_shaped.shape: {train_ys_shaped.shape}')

batch_y0, batch_t, batch_y = ode_batch()
print(f'by0.shape : {batch_y0.shape}, bt.shape: {batch_t.shape}, by.shape: {batch_y.shape}')
print(f'by0.dtype : {batch_y0.dtype}, bt.dtype: {batch_t.dtype}, by.dtype: {batch_y.dtype}')


for epoch in range(1, epochs + 1):
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = ode_batch()
        pred_f = odeint(odefunc, batch_y0, batch_t)  #  , rtol=r_tol, atol=a_tol)
        pred_f = pred_f.view(batch_y.shape)
        # loss = torch.abs(torch.sum(pred_f - batch_y))
        loss = criterion(pred_f, batch_y)
        # loss = torch.mean(torch.abs(pred_f - batch_y))  # loss fxn from ricky
        loss.backward()
        optimizer.step()
        if itr % test_freq == 0:
            with torch.no_grad():
                pred_interpolation = odeint(odefunc, y_0, eval_times)  # , rtol=r_tol, atol=a_tol)
                pred_f = odeint(odefunc, y_0, times)  #  , rtol=r_tol, atol=a_tol)
                # loss = torch.mean(torch.abs(pred_f - ys))
                loss = criterion(pred_f, ys)
                losses.append(loss)
                print('Epoch {} | Iter {:04d} | Total Loss {:.6f}'.format(epoch, itr, loss.item()))
                if viz:
                    visualize(pred_interpolation.squeeze())
                ii += 1

if viz_at_end:
    print(losses)
    loss_over_time = [i for i in range(len(losses))]
    plt.plot(loss_over_time, losses)
    plt.show()
    plt.pause(5)
    plt.ioff()
