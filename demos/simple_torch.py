import argparse

import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim

import matplotlib.pyplot as plt

import loaders as l


class Net(nn.Module):
	def __init__(self, input_size, output_size):
		super(Net, self).__init__()
		self.l1 = nn.Linear(input_size * output_size, 1000)
		self.l2 = nn.Linear(1000, 100)
		self.l3 = nn.Linear(100, 14)
		self.sm = nn.Softmax(dim=0)

	def forward(self, x):
		x = self.l1(x)
		x = F.relu(self.l2(x))
		# x = self.sm(self.l3(x))
		x = self.l3(x)
		x = self.sm(x)
		return x

    
def train(args, model, device, train_loader, optimizer, epoch, criterion):

    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.flatten()
        target = target.flatten()

        # print('data: {}'.format(data))
        # print('target: {}'.format(target))

        optimizer.zero_grad()

        output = model(data)

        # print('output: {}'.format(output))

        loss = criterion(output, target)
        
        # print('loss: {}'.format(loss.detach()))

        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            print('model out: {}'.format(output))
            print('target: {}'.format(target))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main(train_set=l.LSST()):
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--batch-size', type=int, default=64, metavar='N',
	                    help='input batch size for training (default: 64)')
	parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
	                    help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=1, metavar='N',
	                    help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
	                    help='learning rate (default: 0.01)')
	parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
	                    help='SGD momentum (default: 0.5)')
	parser.add_argument('--no-cuda', action='store_true', default=True,
	                    help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=1, metavar='S',
	                    help='random seed (default: 1)')
	parser.add_argument('--log-interval', type=int, default=10, metavar='N',
	                    help='how many batches to wait before logging training status')

	parser.add_argument('--save-model', action='store_true', default=False,
	                    help='For Saving the current Model')
	args = parser.parse_args()
	use_cuda = not args.no_cuda and torch.cuda.is_available()

	torch.manual_seed(args.seed)

	device = torch.device("cuda" if use_cuda else "cpu")

	# train_set = l.LSST()
	train_loader = DataLoader(train_set)

	input_size = train_set.input_shape
	output_size = train_set.output_shape + 1

	print('input_size [0]: {}'.format(input_size))
	print('output_size [0]: {}'.format(output_size))

	criterion = nn.MSELoss()

	model = Net(352, output_size).to(device)
	print(model)
	
	optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

	for epoch in range(1, args.epochs + 1):
		train(args, model, device, train_loader, optimizer, epoch, criterion)
		# test(args, model, device, test_loader)


if __name__ == '__main__':
	main()
