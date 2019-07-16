# huge thanks to https://github.com/EmilienDupont/augmented-neural-odes

import torch
from cosmoNODE.anode.models import ODENet
from cosmoNODE.anode.conv_models import ConvODENet
from cosmoNODE.anode.training import Trainer

from cosmoNODE import loaders as l


'''
This demo file is implementing augmented neural differential equations,
which alter the topology of the inputs by increasing dimensionality
in order to better evaluate the ODE.
'''

# data_loader = l.NDim()
data_loader = l.FluxLoader('training_set')

obj = data_loader.__getitem__(0)
print(obj[0].shape)
print(obj[1])
# print(f'times.head() {obj[0].head()}. fluxes.head() {obj[1].head()}')

device = torch.device('cuda:' if torch.cuda.is_available() else 'cpu')

# # Instantiate a model
# # For regular data...

# are the two dimensions going to be flux and time?

anode = ODENet(device, data_dim=2, hidden_dim=16,
        output_dim=1, augment_dim=1).double()

optimizer = torch.optim.Adam(anode.parameters(), lr=1e-3)
trainer = Trainer(anode, optimizer, device)

# Train model on your dataloader
trainer.train(data_loader, num_epochs=1)