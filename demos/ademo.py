# huge thanks to https://github.com/EmilienDupont/augmented-neural-odes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cosmoNODE.anode.models import ODENet
from cosmoNODE.anode.conv_models import ConvODENet
from cosmoNODE.anode.training import Trainer

# from cosmoNODE import loaders as l
from cosmoNODE.loaders import Anode as A

'''
This demo file is implementing augmented neural differential equations,
which alter the topology of the inputs by increasing dimensionality
in order to better evaluate the ODE.
'''
if __name__ == '__main__':
    # data_loader = l.NDim()
    # data_loader = l.FluxLoader('training_set')
    data_set = A(conv=False)
    obj = data_set[0]

    print(obj[0].shape)

    data_loader = DataLoader(data_set, batch_size=16, shuffle=True)

    # print(f'times.head() {obj[0].head()}. fluxes.head() {obj[1].head()}')

    device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')

    anode = ODENet(device, data_dim=704, hidden_dim=100,
            output_dim=14, augment_dim=10)

    optimizer = torch.optim.Adam(anode.parameters(), lr=1e-3)
    trainer = Trainer(anode, optimizer, device, classification=True, save_dir=('demos/ode_models', '/light_curve0'))

    # Train model on your dataloader
    trainer.train(data_loader, num_epochs=10)
