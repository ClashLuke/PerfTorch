import collections
import time
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from perftorch.locoprop import ConvergenceError, Loco


class LayerNorm2d(nn.LayerNorm):
    def forward(self, input):
        return super(LayerNorm2d, self).forward(input.transpose(1, 3)).transpose(1, 3)


class ConvBlock(nn.Sequential):
    def __init__(self, features: int):
        super().__init__()
        self.add_module("inp_norm", LayerNorm2d(features))
        self.add_module("input", nn.Conv2d(features, features, 3, 1, padding=1))
        self.add_module("relu", nn.ReLU())
        self.add_module("mid_norm", LayerNorm2d(features))
        self.add_module("output", nn.Conv2d(features, features, 3, 1, padding=1))
        self.add_module("out_norm", LayerNorm2d(features))


class GlobalAvgPool(nn.Module):
    def forward(self, inp: torch.Tensor):
        return inp.mean(dim=(2, 3))


class Net(nn.Sequential):
    def __init__(self, feature_factor: int, iterations, lr):
        super(Net, self).__init__()
        self.add_module("input", nn.Conv2d(1, feature_factor, 3, 2))
        for i in range(3):
            if i != 0:
                self.add_module(f"transfer{i}", nn.MaxPool2d(2))
            self.add_module(f"stage{i}", Loco(ConvBlock(feature_factor), iterations, lr))
        self.add_module("pool", GlobalAvgPool())
        self.add_module("classifier", nn.Linear(feature_factor, 10))
        self.add_module("output", nn.LogSoftmax(dim=-1))


def train(model: Net, device: torch.device, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
          locoprop_lr: float, locoprop_iter: int, inner_optimizer: type, learning_rate: float) -> float:
    model.train()
    global_loss = 0

    for batch_idx, (data, target) in enumerate(tqdm.tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        loss = F.nll_loss(model(data), target)
        loss.backward()
        global_loss += loss.detach()
        optimizer.step()
        optimizer.zero_grad()
    return global_loss.item()


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


def get_model(feature_factor: int, device: typing.Union[str, torch.device], iterations, lr) -> Net:
    return Net(feature_factor, iterations, lr).to(device)


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
            locoprop_iter: int, learning_rate: float, inner_optimizer: type, outer_optimizer: type
            ) -> typing.Iterable[typing.Tuple[float, float]]:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = get_model(feature_factor, device, locoprop_iter, learning_rate)
    optimizer: torch.optim.Optimizer = outer_optimizer(model.parameters(), lr=learning_rate)

    train_loader = get_dataset(batch_size, True)
    test_loader = get_dataset(16_384, False)

    for _ in range(epochs):
        start_time = time.time()
        train(model, device, train_loader, optimizer, locoprop_lr, locoprop_iter, inner_optimizer, learning_rate)
        out = test(model, device, test_loader)
        yield out, time.time() - start_time
    if use_cuda:
        torch.cuda.empty_cache()


def main():
    kwargs = {"seed": 0, "feature_factor": 16, "batch_size": 2048, "epochs": 16,
              "outer_optimizer": torch.optim.SGD, "inner_optimizer": torch.optim.SGD,
              "locoprop_lr": 1e-4  # ignored
              }

    baseline_losses = [run_one(locoprop_iter=itr, learning_rate=lr, **kwargs)
                       for lr in [1e-4, 1e-6, 1e-8]
                       for itr in [1, 256]]
    while True:
        try:
            for base in baseline_losses:
                try:
                    loss, took = next(base)
                except StopIteration:
                    print("Skipped")
                    continue
                except ConvergenceError as exc:
                    print(exc)
                    continue
                print(f"Loss: {loss} - Took: {took:.1f}s")
        except StopIteration:
            return
        print("-" * 12)


if __name__ == '__main__':
    main()
