from __future__ import print_function

import collections
import os
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

EPOCHS = 16


class MulFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, scale: torch.Tensor):
        ctx.save_for_backward(inp, scale)
        return inp * scale.view(1, -1, 1, 1)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        if not ctx.saved_tensors:
            return None, None, None, None
        inp, scale = ctx.saved_tensors
        scale_grad = (dy * inp).sum((0, 2, 3)).reshape(-1)
        scale.square_grad = (dy * inp).square().sum((0, 2, 3)).reshape(-1) * inp.size(0)
        return dy * scale.view(1, -1, 1, 1), scale_grad


class LayerNorm(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        return MulFn.apply((x - x.mean(1, True)) / x.norm(2, 1, True) * x.size(1) ** 0.5, self.scale)


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
        inp = F.relu(inp)
        if self.use_square:
            return LinearFn.apply(inp, self.weight)
        else:
            return inp @ self.weight


class ConvBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_square: bool, stride: int, normalization: type,
                 residual: bool, activate: bool):
        super().__init__()
        if activate:
            self.normalization = normalization(in_features)
        self.activate = activate
        self.unfold = nn.Unfold(kernel_size=(3, 3), stride=(stride, stride), padding=(0, 0))
        self.linear = Linear(in_features * 9, out_features, use_square)
        self.out_features = out_features
        self.residual = residual
        self.stride = stride

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        batch, _, width, _ = inp.size()
        if self.activate:
            out = F.relu(self.normalization(inp))
        else:
            out = inp
        out = self.unfold(out).transpose(1, 2)
        out = self.linear(out).transpose(1, 2)
        size = out.size(-1)
        out = out.contiguous().reshape(batch, self.out_features, int(size ** 0.5), int(size ** 0.5))
        if self.residual:
            inp = inp[:, :, 1:-1:self.stride, 1:-1:self.stride]
            if inp.size(1) != out.size(1):
                inp = torch.cat([inp, torch.zeros((inp.size(0), out.size(1) - inp.size(1), inp.size(2), inp.size(3)),
                                                  dtype=inp.dtype, device=inp.device)], 1)
            out = out + inp
        return out


class LinearBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, drop: float, use_square: bool):
        super().__init__()
        self.drop = nn.Dropout(drop)  # we can't wrap stochastic things like dropout as RNG isn't kept
        self.lin = Linear(in_features, out_features, use_square)

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        return self.lin(self.drop(inp))


class Net(nn.Module):
    def __init__(self, feature_factor: int, use_square: bool, depth: int, classes: int, dropout: float,
                 normalization: type, residual: bool, input_size: int):
        super(Net, self).__init__()
        self.conv1 = ConvBlock(1 if input_size == 28 else 3, feature_factor, use_square, 2, normalization, residual,
                               False)
        self.stem = nn.Sequential(
                *(ConvBlock(feature_factor, feature_factor, use_square, 1, normalization, residual, True) for _ in
                  range(depth)))
        self.conv2 = ConvBlock(feature_factor, feature_factor * 2, use_square, 2, normalization, residual, True)
        self.fc1 = LinearBlock((((input_size - 3) // 2 - 2 * (1 + depth)) // 2 + 1) ** 2 * feature_factor * 2,
                               feature_factor * 4, dropout, use_square)
        self.fc2 = LinearBlock(feature_factor * 4, classes, dropout, use_square)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stem(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        return self.fc2(self.fc1(x))


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


def get_dataset(batch_size: int, training: bool, dataset: str) -> DataLoader:
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = getattr(datasets, dataset)('../data', train=training, download=True, transform=transform)
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
    def __init__(self, *args, graft: bool = False, beta3: typing.Optional[float] = None, **kwargs):
        super(AdamW, self).__init__(*args, **kwargs)
        self.graft = graft
        self.beta3 = beta3

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta3 = self.beta3 if self.beta3 is not None else beta2

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = torch.tensor(0.)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_true_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if self.graft:
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state['exp_avg']
                exp_avg_true_sq = state['exp_avg_true_sq']
                step_t = state['step']

                # update step
                step_t += 1

                # Perform stepweight decay
                p.mul_(1 - group['lr'] * group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
                exp_avg_true_sq.mul_(beta3).add_(p.square_grad, alpha=1 - beta3)

                step = step_t.item()

                # sqrt(sum(denom0^2)/sum(denom1^2)), but denom^2*const == exp_avg_sq -> sqrt(sum(x)/sum(y))
                if self.graft:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(p.grad.square(), alpha=1 - beta2)
                    scale = exp_avg_sq.sum() / (exp_avg_true_sq.sum() + 1e-12)
                else:
                    scale = 1

                denom = (exp_avg_true_sq / (1 - beta3 ** step)).sqrt().add_(group['eps'])
                p.addcdiv_(exp_avg, denom, value=-group['lr'] / (1 - beta1 ** step) * scale)


@generator_cache
def run_one(seed: int, feature_factor: int, batch_size: int, learning_rate: float, use_square: bool, depth: int,
            classes: int, dataset: str, dropout: float, normalization: type, residual: bool, input_size: int,
            graft: bool, beta1: float, beta2: float, beta3: float
            ) -> typing.Iterable[float]:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(feature_factor, use_square, depth, classes, dropout, normalization, residual, input_size).to(device)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")
    if use_square:
        optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2), beta3=beta3, graft=graft,
                          eps=1e-12)
    else:
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2),
                                                             eps=1e-12)

    train_loader = get_dataset(batch_size, True, dataset)
    test_loader = get_dataset(16_384, False, dataset)

    for _ in range(EPOCHS):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer)
        test_loss, test_accuracy = test(model, device, test_loader)
        yield train_loss, train_accuracy, test_loss, test_accuracy
    if use_cuda:
        torch.cuda.empty_cache()


def log_one(cfg):
    if not cfg["use_square"] and cfg["graft"]:
        return
    if cfg["use_square"] and not cfg["graft"] and cfg["beta2"] != cfg["beta3"]:
        return
    wandb.init(project="truegrad-varying-arch", entity="clashluke", reinit=True, config=cfg)
    run = run_one(**cfg)
    best_te_acc = 0
    for tr_loss, tr_acc, te_loss, te_acc in run:
        best_te_acc = max(best_te_acc, te_acc)
        wandb.log({"Train Loss": tr_loss, "Train Accuracy": tr_acc, "Test Loss": te_loss, "Test Accuracy": te_acc,
                   "Best Test Accuracy": best_te_acc
                   })

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
               "graft": [True, False],
               "seed": [0, 1],
               "batch_size": [512, 16384],
               "feature_factor": [8, 32, 128],
               "learning_rate": [1e-2, 1e-4],
               "beta1": [0.8, 0.9, 0.95],
               "beta2": [0.95, 0.99, 0.999],
               "beta3": [0.95, 0.99, 0.999],
               "dropout": [0, 0.2],
               "normalization": [nn.Identity, nn.InstanceNorm2d, LayerNorm],
               "residual": [True, False],
               "depth": [0, 3],
               "dataset": ["MNIST", "CIFAR10", "CIFAR100"]
               }
    configs = list(product(options))
    for cfg in tqdm.tqdm(configs):
        cfg["classes"] = 100
        cfg["input_size"] = 32
        if cfg["dataset"] == "MNIST":
            cfg["input_size"] = 28
        if cfg["dataset"] == "CIFAR100":
            cfg["classes"] = 10
        for i in range(4):
            try:
                log_one(cfg)
                break
            except RuntimeError:
                traceback.print_exc()


if __name__ == '__main__':
    log_all()
