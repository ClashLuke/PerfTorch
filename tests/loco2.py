import collections
import time
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from perftorch.locoprop import ConvergenceError, Loco2


class LayerNorm2d(nn.LayerNorm):
    def forward(self, input):
        return super(LayerNorm2d, self).forward(input.transpose(1, 3)).transpose(1, 3)


class ConvBlock(nn.Sequential):
    def __init__(self, features: int, stride: int):
        super().__init__()
        self.add_module("relu", nn.ReLU())
        self.add_module("norm", LayerNorm2d(features))
        self.add_module("input", nn.Conv2d(features, features, 3, stride, padding=1))


class GlobalAvgPool(nn.Module):
    def forward(self, inp: torch.Tensor):
        return inp.mean(dim=(2, 3))


class Head(nn.Sequential):
    def __init__(self, feature_factor: int):
        super(Head, self).__init__()
        self.add_module("pool", GlobalAvgPool())
        self.add_module("classifier", nn.Linear(feature_factor, 10))


class Net(nn.Sequential):
    def __init__(self, feature_factor: int):
        super(Net, self).__init__()
        self.add_module("input", nn.Conv2d(1, feature_factor, 3, 2))
        for i in range(6):
            self.add_module(f"stage{i}", ConvBlock(feature_factor, 1 + int(i > 0 and i % 2)))
        self.add_module("classifier", Head(feature_factor))


def train(model: Net, device: torch.device, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
          locoprop_lr: float, locoprop_iter: int, inner_optimizer: type, learning_rate: float) -> float:
    model.train()
    global_loss = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        loss = F.cross_entropy(model(data), target)
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
            test_loss += F.cross_entropy(output, target, reduction='sum').detach()

    test_loss /= len(test_loader.dataset)
    return test_loss.item()


def get_model(feature_factor: int, device: typing.Union[str, torch.device], iterations, lr) -> Net:
    return Loco2(Net(feature_factor), iterations, lr).to(device)


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
    kwargs = {"seed": 0, "feature_factor": 16, "epochs": 128,
              "outer_optimizer": torch.optim.SGD, "inner_optimizer": torch.optim.SGD,
              "locoprop_lr": 1e-4  # ignored
              }

    losses = {f"[LR={lr:9.6f}][LocalIterations={itr:1d}][Batch={batch:5d}]":
                  run_one(batch_size=batch, locoprop_iter=itr, learning_rate=lr, **kwargs)
              for lr in [3 ** -i for i in range(2, 9)] for itr in range(3) for batch in [4 ** i for i in range(3, 8)]}

    for i in range(kwargs["epochs"]):
        for name, base in losses.items():
            try:
                loss, took = next(base)
            except StopIteration:
                print(f"[Epoch={i}]{name} Skipped")
                continue
            except ConvergenceError as exc:
                print(f'[Epoch={i}]{name} {exc}')
                continue
            if loss > 10:
                print(f"[Epoch={i}]{name} diverged")
                losses[name] = iter([])  # -> raise StopIteration in next loop call
                continue

            print(f"[Epoch={i}]{name} Loss: {loss:8.6f} - Took: {took:4.1f}s")
        print("-" * 12)


if __name__ == '__main__':
    main()
