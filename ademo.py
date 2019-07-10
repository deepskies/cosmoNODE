# huge thanks to https://github.com/EmilienDupont/augmented-neural-odes

import torch
from anode.models import ODENet
from anode.conv_models import ConvODENet
from anode.training import Trainer

import loaders as l


'''
This demo file is implementing augmented neural differential equations,
which alter the topology of the inputs by increasing dimensionality
in order to better evaluate the ODE.
'''

data_loader = l.FluxLoader('training_set')
obj = data_loader.__getitem__(0)
print(f'times.head() {obj[0].head()}. fluxes.head() {obj[1].head()}')

device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')

# # Instantiate a model
# # For regular data...
# anode = ODENet(device, data_dim=2, hidden_dim=16, augment_dim=1)
# # ... or for images
# anode = ConvODENet(device, img_size=(1, 28, 28), num_filters=32, augment_dim=1)
#
# # Instantiate an optimizer and a trainer
# optimizer = torch.optim.Adam(anode.parameters(), lr=1e-3)
# trainer = Trainer(anode, optimizer, device)
#
# # Train model on your dataloader
# trainer.train(dataloader, num_epochs=10)
