import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

# simple multilayer linear model for mnist
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = nn.Linear(784, 392)
        self.l2 = nn.Linear(392, 196)
        self.l3 = nn.Linear(196, 98)
        self.l4 = nn.Linear(98, 49)
        self.l5 = nn.Linear(49, 10)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        x = F.relu(self.l5(x))
        return F.log_softmax(x, dim=1)


#todo
"""
* epoch loss
* rms loss (moving average)
* plot
"""

if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data/', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=1, shuffle=True)

    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    epochs = 10
    correct = 0
    tot = 0
    length = len(train_loader)
    log_freq = 20
    for epoch in range(1, epochs + 1):
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            x, y = data
            x = x.reshape(1, -1)
            y_pred = model(x)
            loss = F.nll_loss(y_pred, y)
            loss.backward()
            optimizer.step()
            tot += 1

            pred = y_pred.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(y.view_as(pred)).sum().item()

            acc = correct / tot

            if tot % log_freq == 0:
                print(f'total Accuracy: {acc}, {correct} / {tot}')
