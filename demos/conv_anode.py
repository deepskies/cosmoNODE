# classification using augmented neural ODEs
import torch
from cosmoNODE.anode.conv_models import ConvODENet
from cosmoNODE.anode.models import ODENet
from cosmoNODE.anode.training import Trainer

from cosmoNODE.loaders import Anode as A

if __name__ == '__main__':
    device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')
    conv_net = ConvODENet(device, img_size=(1, 352, 2), num_filters=32, augment_dim=1).double()
    dataloader = A()
    optimizer = torch.optim.Adam(conv_net.parameters(), lr=1e-3)
    trainer = Trainer(conv_net, optimizer, device)
    trainer.train(dataloader, num_epochs=1)
