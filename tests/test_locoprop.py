"""
Base code taken from https://github.com/pytorch/examples/blob/main/mnist/main.py
"""
from __future__ import print_function

import collections
import time
import typing

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from perftorch.locoprop import LocopropWrapper


class ConvBlock(nn.Module):  # wrapping conv + relu == locoprop-m
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.inp = nn.Conv2d(in_features, out_features, 3, 2)
        self.out = nn.Conv2d(out_features, out_features, 3, 1)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.inp(inp)))


class Net(nn.Module):
    def __init__(self, feature_factor: int = 1):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(1, feature_factor)
        self.conv2 = ConvBlock(feature_factor, 10)
        # self.fc1 = LinearBlock(3 * 3 * feature_factor * 2, feature_factor * 4, 0.25)
        # self.fc2 = LinearBlock(feature_factor * 4, 10, 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.mean(dim=(2, 3))
        return F.log_softmax(x, dim=1)


def train(model: Net, device: torch.device, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    model.train()
    global_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = F.nll_loss(output, target, reduction='mean')
        global_loss += loss.detach()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return global_loss.item()


@pytest.mark.skip
def test(model: Net, device: torch.device, test_loader: DataLoader) -> float:
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').detach()

    test_loss /= len(test_loader.dataset)
    return test_loss.item()


def get_model(feature_factor: int, device: typing.Union[str, torch.device], locoprop_lr: float,
              locoprop_iter: int) -> Net:
    model = Net(feature_factor).to(device)

    locoprop_args = [locoprop_lr, 1, locoprop_iter]
    model.conv1 = LocopropWrapper(model.conv1, *locoprop_args)
    model.conv2 = LocopropWrapper(model.conv2, *locoprop_args)
    # model.fc1.lin = LocopropWrapper(model.fc1.lin, *locoprop_args)
    # model.fc2.lin = LocopropWrapper(model.fc2.lin, *locoprop_args)

    return model


def get_dataset(batch_size: int, training: bool) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST('../data', train=training, download=True, transform=transform)
    kwargs = {'num_workers': 8, 'pin_memory': True, 'shuffle': True} if torch.cuda.is_available() else {}
    return DataLoader(dataset, batch_size=batch_size, **kwargs)


def generator_cache(fn: typing.Callable):
    cache = collections.defaultdict(list)

    def _fn(*args, **kwargs):
        name = str(args) + str(kwargs)
        if name in cache:
            for idx in cache[name]:
                yield idx
            for _, _ in zip(range(len(cache[name])), fn(*args, **kwargs)):
                pass
        for out in fn(*args, **kwargs):
            cache[name].append(out)
            yield out

    return _fn


@generator_cache
def run_one(seed: int, feature_factor: int, batch_size: int, epochs: int, locoprop_lr: float,
            locoprop_iter: int, optimizer: type, learning_rate: float
            ) -> typing.Iterable[float]:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = get_model(feature_factor, device, locoprop_lr, locoprop_iter)
    optimizer: torch.optim.Optimizer = optimizer(model.parameters(), lr=learning_rate)

    train_loader = get_dataset(batch_size, True)
    test_loader = get_dataset(16_384, False)

    for _ in range(epochs):
        start_time = time.time()
        train(model, device, train_loader, optimizer)
        end_time = time.time()
        yield test(model, device, test_loader), end_time - start_time

    if use_cuda:
        torch.cuda.empty_cache()


@pytest.mark.parametrize("batch_size", [1024])
@pytest.mark.parametrize("locoprop_lr", [1, 0.01])
@pytest.mark.parametrize("locoprop_iter", [10])
@pytest.mark.parametrize("feature_factor", [32])
@pytest.mark.parametrize("learning_rate", [0.001])
@pytest.mark.parametrize("optimizer", [torch.optim.SGD, torch.optim.AdamW, torch.optim.RMSprop])
@pytest.mark.parametrize("seed", [0])
def test_main(seed: int, feature_factor: int, batch_size: int, locoprop_lr: float, locoprop_iter: int,
              learning_rate: float, optimizer: type, epochs: int = 8):
    kwargs = {"seed": seed, "feature_factor": feature_factor, "batch_size": batch_size, "epochs": epochs,
              "locoprop_lr": locoprop_lr, "optimizer": optimizer, "learning_rate": learning_rate
              }
    epoch = 1
    try:
        baseline_losses = run_one(locoprop_iter=1, **kwargs)
        locoprop_losses = run_one(locoprop_iter=locoprop_iter, **kwargs)
        for baseline, locoprop in zip(baseline_losses, locoprop_losses):
            msg = f"Baseline (test) better than LocoProp @ Epoch {int(epoch)}"
            assert baseline >= locoprop, msg
            epoch += 1
    except RuntimeError:
        pytest.skip("OOM")


def main():
    kwargs = {"seed": 0, "feature_factor": 32, "epochs": 16, "optimizer": torch.optim.SGD}

    best_loss = float("inf")
    best_config = [-1, -1, -1, -1]
    with open("log.txt", 'w') as f:
        for batch in range(7, 11, 3):
            batch = 2 ** batch
            for lr in range(1, 8, 3):
                lr = 3 ** -lr
                for itr in range(0, 3, 2):
                    itr = 2 ** itr
                    for (loss, took), epoch in zip(run_one(batch_size=batch, locoprop_iter=itr, learning_rate=lr,
                                                           locoprop_lr=10000, **kwargs),
                                                   range(1, 1 + 4)):
                        if loss < best_loss:
                            best_loss = loss
                            best_config = [itr, lr, epoch, batch]
                        msg = (f"[{batch}][{lr:8f}][{itr:2d}][{epoch:2d}] | Loss: {loss:8.5f} - Took: {took:4.1f}s | "
                               f"Best Loss: {best_loss:10.8f} - Iterations: {best_config[0]} - LR: {best_config[1]} - "
                               f"Epoch: {best_config[2]} - Batch: {best_config[3]}")
                        print(msg)
                        f.write(msg + '\n')
                    torch.cuda.empty_cache()
            print('--------')
        print('\n--------\n')
    print('\n\n\n--------\n\n\n')


if __name__ == '__main__':
    main()
