# classification using augmented neural ODEs
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from cosmoNODE.anode.conv_models import ConvODENet
from cosmoNODE.anode.models import ODENet
from cosmoNODE.anode.training import Trainer

from cosmoNODE.loaders import Anode as A

if __name__ == '__main__':
    # dataloader = DataLoader(datasets.MNIST('./demos/data', train=True, transform=transforms.ToTensor()), batch_size=64)
    a = A()
    dataloader = DataLoader(a, batch_size=16)

    device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
    conv_net = ConvODENet(device, img_size=(1, 352, 2), num_filters=32, output_dim=14, augment_dim=1)
    # conv_net = ConvODENet(device, img_size=(1, 352, 2), num_filters=32, augment_dim=1).double()
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=1e-2)
    trainer = Trainer(conv_net, optimizer, device, classification=True, save_dir=('demos/ode_models', '/light_curve0'))
    # trainer = Trainer(conv_net, optimizer, device, classification=True)
    trainer.train(dataloader, num_epochs=10)
