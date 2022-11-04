"""
Base code taken from https://github.com/pytorch/examples/blob/main/mnist/main.py
"""
from __future__ import print_function
import os
import collections
import random
import traceback
import typing

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

os.environ["WANDB_SILENT"] = "true"

import wandb


class LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(inp, weight)
        return inp @ weight

    @staticmethod
    def backward(ctx, dy: torch.Tensor) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        inp, wgt = ctx.saved_tensors
        lhs = ''.join(chr(ord('a') + i) for i in range(dy.ndim - 1))
        d_wgt = torch.einsum(f"{lhs}y,{lhs}z->yz", inp, dy)
        d_wgt_sq = torch.einsum(f"{lhs}y,{lhs}z->yz", inp.square(), dy.square() * inp.size(0))  # * size since mean
        wgt.square_grad = d_wgt_sq
        d_inp = torch.einsum(f"{lhs}z,yz->{lhs}y", dy, wgt)
        return d_inp, d_wgt


class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_square: bool) -> None:
        super(Linear, self).__init__()
        self.weight = torch.nn.Parameter(torch.randn((in_features, out_features)) / in_features ** 0.5)
        self.use_square = use_square

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        if self.use_square:
            return LinearFn.apply(inp, self.weight)
        else:
            return inp @ self.weight


class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_square: bool):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.linear = Linear(in_features * 9, out_features, use_square)
        self.out_features = out_features

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        batch, _, width, _ = inp.size()
        inp = self.unfold(inp).transpose(1, 2)
        inp = self.linear(inp).transpose(1, 2)
        size = inp.size(-1)
        return F.relu(inp).reshape(batch, self.out_features, int(size ** 0.5), int(size ** 0.5))


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, drop: float, use_square: bool):
        super().__init__()
        self.drop = nn.Dropout(drop)  # we can't wrap stochastic things like dropout as RNG isn't kept
        self.lin = Linear(in_features, out_features, use_square)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.lin(self.drop(inp))


class Net(nn.Module):
    def __init__(self, feature_factor: int = 1, use_square: bool = True):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(1, feature_factor, use_square)
        self.conv2 = ConvBlock(feature_factor, feature_factor * 2, use_square)
        self.fc1 = LinearBlock(3 * 3 * feature_factor * 2, feature_factor * 4, 0.25, use_square)
        self.fc2 = LinearBlock(feature_factor * 4, 10, 0.5, use_square)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        return self.fc2(x)


def train(model: Net, device: torch.device, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    global_loss = 0
    accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        accuracy += (torch.argmax(output, dim=1) == target).sum().detach()
        global_loss += loss.detach() * data.size(0)
        loss.backward()
        optimizer.step()
    global_loss /= len(train_loader.dataset)
    accuracy = accuracy.float()
    accuracy /= len(train_loader.dataset)
    return global_loss.item(), accuracy.item()


@pytest.mark.skip
def test(model: Net, device: torch.device, test_loader: DataLoader):
    model.eval()
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += F.cross_entropy(output, target, reduction='sum').detach()
            accuracy += (torch.argmax(output, dim=1) == target).sum().detach()

    loss /= len(test_loader.dataset)
    accuracy = accuracy.float()
    accuracy /= len(test_loader.dataset)
    return loss.item(), accuracy.item()


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


class AdamW(torch.optim.AdamW):
    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                step_t = state['step']

                # update step
                step_t += 1

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).add_(p.square_grad, alpha=1 - beta2)

                step = step_t.item()

                denom = (exp_avg_sq / (1 - beta2 ** step)).sqrt().add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-group['lr'] / (1 - beta1 ** step))


@generator_cache
def run_one(seed: int, feature_factor: int, batch_size: int, epochs: int, learning_rate: float, use_square: bool
            ) -> typing.Iterable[float]:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(feature_factor, use_square).to(device)
    if use_square:
        optimizer = AdamW(model.parameters(), lr=learning_rate)
    else:
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    train_loader = get_dataset(batch_size, True)
    test_loader = get_dataset(16_384, False)

    for _ in range(epochs):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer)
        test_loss, test_accuracy = test(model, device, test_loader)
        yield train_loss, train_accuracy, test_loss, test_accuracy
    if use_cuda:
        torch.cuda.empty_cache()


def log_one(cfg, seed: int, feature_factor: int, batch_size: int, learning_rate: float, use_square: bool,
            epochs: int = 16):
    wandb.init(project="truegrad", entity="clashluke", reinit=True, config=cfg)
    run = run_one(seed=seed, feature_factor=feature_factor, batch_size=batch_size, epochs=epochs,
                  learning_rate=learning_rate, use_square=use_square)
    for tr_loss, tr_acc, te_loss, te_acc in run:
        wandb.log({"Train Loss": tr_loss, "Train Accuracy": tr_acc, "Test Loss": te_loss, "Test Accuracy": te_acc})

    wandb.finish()


def product(x):
    key = next(iter(x.keys()))
    itm = x.pop(key)
    if x:
        for item in product(x):
            for val in itm:
                item = item.copy()
                item[key] = val
                yield item
    else:
        for val in itm:
            yield {key: val}


def log_all():
    options = {"use_square": [True, False],
               "seed": [0, 1],
               "batch_size": [32, 128, 512, 2048, 8192][::-1],
               "feature_factor": [4, 16, 64],
               "learning_rate": [0.1, 0.01, 0.001],
               }
    configs = list(product(options))
    for cfg in tqdm.tqdm(configs):
        for i in range(4):
            try:
                log_one(cfg, **cfg)
                break
            except RuntimeError:
                traceback.print_exc()


if __name__ == '__main__':
    log_all()
