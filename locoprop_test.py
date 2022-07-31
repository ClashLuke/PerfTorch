from __future__ import print_function

import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from locoprop_wrapper import LocopropWrapper


class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, 3, 2)

    def forward(self, inp: torch.Tensor):
        return F.relu(self.conv(inp))


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, drop: float):
        super().__init__()
        self.drop = nn.Dropout(drop)
        self.lin = nn.Linear(in_features, out_features)

    def forward(self, inp: torch.Tensor):
        return self.lin(self.drop(inp))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(1, 32)
        self.conv2 = ConvBlock(32, 64)
        self.fc1 = LinearBlock(3 * 3 * 64, 128, 0.25)
        self.fc2 = LinearBlock(128, 10, 0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    start_time = time.time()
    loss_mean = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss_mean += loss.detach()
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            step_time = (batch_idx + 1) / (time.time() - start_time)
            print(f'Train Epoch: {epoch} [{batch_idx * len(data):5d}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss_mean.item() / args.log_interval:.6f} | '
                  f'Rate: {step_time:6.1f} step/s - {step_time * len(data):7.0f} img/s')
            loss_mean = 0


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument("--locoprop", type=int, default=1, help="Whether to use LocoProp (0/1)")
    parser.add_argument('--batch-size', type=int, default=2048, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=16384, metavar='N',
                        help='input batch size for testing (default: 1024)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=16, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)

    if args.locoprop:
        locoprop_args = [args.lr, 1, 10]
        model.conv1 = LocopropWrapper(model.conv1, *locoprop_args)
        model.conv2 = LocopropWrapper(model.conv2, *locoprop_args)
        model.fc1.lin = LocopropWrapper(model.fc1.lin, *locoprop_args)
        model.fc2.lin = LocopropWrapper(model.fc2.lin, *locoprop_args)

    optimizer = optim.SGD(model.parameters(), lr=1)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()
