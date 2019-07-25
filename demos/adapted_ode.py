import matplotlib.pyplot as plt
# %matplotlib inline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchdiffeq import odeint

from cosmoNODE.loaders import Anode as A

device = torch.device('cpu')

''' this is an adapted version of ricky's ode_demo.py '''

viz = True
data_size = 350
batch_time = 10
batch_size = 20
test_freq = 20

a = A()
dataloader = DataLoader(a)
item = a[0]
lc = item[0]
obj_class = item[1]
lc = lc.squeeze()

mjds = lc[:, 0]
fluxes = lc[:, 1]
flux_y0 = fluxes[0].reshape(1, -1) # torch.Size([1, 1])
true_f = fluxes.reshape(-1, 1, 1)

def flux_batch():
    s = torch.from_numpy(np.random.choice(np.arange(data_size - batch_time, dtype=np.int64), batch_size, replace=False))
    batch_y0 = true_f[s]  # (M, D)
    batch_t = mjds[:batch_time]  # (T)
    batch_y = torch.stack([true_f[s + i] for i in range(batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

by0_f, bt_f, by_f = flux_batch()
# print(f'by0.shape : {by0.shape}, bt.shape: {bt.shape}, by.shape: {by.shape}')
print(f'by0.shape : {by0_f.shape}, bt.shape: {bt_f.shape}, by.shape: {by_f.shape}')

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
        return self.net(y)

if viz:
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

def visualize(true_y, pred_y, odefunc, itr):
    ax_traj.cla()
    ax_traj.set_title('Trajectories')
    ax_traj.set_xlabel('mjd')
    ax_traj.set_ylabel('flux')
    ax_traj.plot(mjds.numpy(), true_y.numpy()[:, 0, 0])
    ax_traj.plot(mjds.numpy(), pred_y.numpy()[:, 0, 0], '--')
    ax_traj.set_xlim(mjds.min(), mjds.max())
    ax_traj.legend()

    fig.tight_layout()
    plt.draw()
    plt.pause(0.001)


if __name__ == '__main__':
    # viz_init()
    niters = 100
    odefunc = ODEFunc()
    optimizer = optim.RMSprop(odefunc.parameters(), lr=1e-3)
    ii = 0
    for itr in range(1, niters + 1):
        optimizer.zero_grad()
        by0_f, bt_f, by_f = flux_batch()
        pred_f = odeint(odefunc, by0_f, bt_f)
        loss = torch.mean(torch.abs(pred_f - by_f))
        loss.backward()
        optimizer.step()
        if itr % test_freq == 0:
            with torch.no_grad():
                pred_f = odeint(odefunc, flux_y0, mjds)
                loss = torch.mean(torch.abs(pred_f - true_f))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_f, pred_f, odefunc, ii)
                ii += 1
