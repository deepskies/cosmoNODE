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


# t = torch.linspace(0., 25., args.data_size) # objects times


true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]])  # ???

def get_batch(itr, item):
    t = item[0]
    true_y = item[1]
    y_len = len(true_y)

    y_cutoff = args.batch_size * itr
    y_lower_cutoff = y_cutoff - args.batch_size

    if y_cutoff >= y_len:
        return None

    s = torch.tensor([i for i in range(y_lower_cutoff, y_cutoff)])

    batch_y0 = true_y[0].reshape(-1, 1)  # read from dataframe

    t_cutoff = args.batch_time * itr
    t_lower_cutoff = t_cutoff - args.batch_time

    batch_t = t[t_lower_cutoff:t_cutoff]  # (T)

    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y

def their_get_batch(item):
    t = item[0]
    true_y = item[1]
    y_len = len(true_y)
    s = torch.from_numpy(np.random.choice(np.arange(y_len - args.batch_time, dtype=np.int64), args.batch_size, replace=False))

    batch_y0 = true_y[s]  # (M, D)

    batch_t = t[:args.batch_time]  # (T)

    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0, batch_t, batch_y


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


if args.viz:
    makedirs('png')
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)


def visualize(true_y, pred_y, odefunc, itr):

    if args.viz:

        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.numpy(), true_y.numpy()[:, 0, 0], t.numpy(), true_y.numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.numpy(), pred_y.numpy()[:, 0, 0], '--', t.numpy(), pred_y.numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.min(), t.max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.numpy()[:, 0, 0], true_y.numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.numpy()[:, 0, 0], pred_y.numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')

        y, x = np.mgrid[-2:2:21j, -2:2:21j]
        dydt = odefunc(0, torch.Tensor(np.stack([x, y], -1).reshape(21 * 21, 2))).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)

        ax_vecfield.streamplot(x, y, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig('png/{:03d}'.format(itr))
        plt.draw()
        plt.pause(0.001)


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.net = self.net.double()

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
    y = item[1]
    return t, y


def test():
    # this works
    t, y = flux_item()

    ii = 0

    true_t0 = t[:args.batch_time] # .reshape([1])
    true_y0 = y[0].reshape([-1, 1])

    func = ODEFunc().double()
    print(func)

    pred_y = odeint(func, true_y0, true_t0).double()

    loss = torch.mean(torch.abs(pred_y - y[:10])).requires_grad_(True)

    print(pred_y)
    print(loss)

    visualize(true_y0, pred_y, func, ii)


if __name__ == '__main__':

    ii = 0

    flux_loader = fl()

    t, y = flux_item()

    seq_len = len(t)
    # print(t)

    true_t0 = t[0].reshape([1])
    true_y0 = y[0].reshape([1])

    print(true_t0, true_y0)

    func = ODEFunc().double()

    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)


    for itr in range(1, args.niters + 1):

        optimizer.zero_grad()
        # batch = their_get_batch((t, y))
        # if batch is None:
        #     print('obj finished')
        #     break

        # batch_y0, batch_t, batch_y = batch
        # batch_y0 = batch_y0.view([-1, 1])
        # true_t0 = t[itr].reshape([1])
        # true_y0 = y[itr].reshape([1])
        up_bound = itr*args.batch_time
        if up_bound >= seq_len:
            break

        batch_t = t[up_bound - args.batch_time:itr*args.batch_time]

        pred_y = odeint(func, true_y0, batch_t)  # going to return batch_time # of predictions

        batch_y = y[up_bound - args.batch_time:itr*args.batch_time]

        # pred_y = odeint(func, true_y0, true_t0).double()
        print(pred_y[1], batch_y[1])

        loss = torch.mean(torch.abs(pred_y - batch_y)).requires_grad_(True)
        # print('real: ({}, {})'.format(batch_t, true_y0))
        # print('pred_y: {}'.format(pred_y))
        print('loss: {}'.format(loss))
        loss.backward()
        optimizer.step()

        if itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y = odeint(func, true_y0, batch_t)
                loss = torch.mean(torch.abs(pred_y - batch_y))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                visualize(true_y0, pred_y, func, ii)
                ii += 1

        end = time.time()
    print('done')
