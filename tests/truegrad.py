import traceback
import typing

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import typer
import wandb
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from truegrad import optim

EPOCHS = 16


def get_dataset(dset: str, training: bool):
    return list(getattr(datasets, dset)('../data', train=training, download=True, transform=transforms.ToTensor()))


DSETS = {dset: [get_dataset(dset, training) for training in [True, False]] for dset in ["MNIST", "CIFAR10", "CIFAR100"]}


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
        scale.sum_grad_squared = (dy * inp).square().sum((0, 2, 3)).reshape(-1) * inp.size(0)
        return dy * scale.view(1, -1, 1, 1), scale_grad


class LayerNorm(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(in_features))

    def forward(self, x):
        x = x - x.mean(1, True)
        return MulFn.apply(x / (1e-8 + x.norm(2, 1, True)) * (x.size(1) - 1) ** 0.5, self.scale)


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
        wgt.sum_grad_squared = d_wgt_sq
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


class NaN(ValueError):
    pass


def train(model: Net, device: torch.device, train_loader: DataLoader, optimizer: torch.optim.Optimizer):
    model.train()
    global_loss = 0
    accuracy = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        if torch.logical_or(torch.isnan(loss), torch.isinf(loss)).item():
            raise NaN
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


def get_dataloader(batch_size: int, training: bool, dataset: str) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True)


normalizations = {"InstanceNorm2d": nn.InstanceNorm2d, "Identity": nn.Identity, "LayerNorm": LayerNorm}


def run_one(seed: int, feature_factor: int, batch_size: int, learning_rate: float, use_square: bool, depth: int,
            dataset: str, dropout: float, normalization: str, residual: bool, graft: bool, beta1: float, beta2: float,
            beta3: float, beta4: float, optimizer: str):
    normalization = normalizations[normalization]
    input_size = 28 if dataset == "MNIST" else 32
    classes = 100 if dataset == "CIFAR100" else 10
    use_cuda = torch.cuda.is_available()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net(feature_factor, use_square, depth, classes, dropout, normalization, residual, input_size).to(device)
    print(model)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,d}")

    optimizer = getattr(optim, optimizer)(model.parameters(), lr=learning_rate, betas=(beta1, beta2, beta3, beta4),
                                          graft=graft, enforce_baseline=not use_square)

    train_dset, test_dset = DSETS[dataset]
    train_loader = DataLoader(train_dset, batch_size=batch_size, num_workers=1, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_dset, batch_size=16_384, num_workers=1, pin_memory=True, shuffle=True)

    best_test_accuracy = 0
    for _ in range(EPOCHS):
        train_loss, train_accuracy = train(model, device, train_loader, optimizer)
        if train_loss > 10:
            wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy,
                       "Best Test Accuracy": best_test_accuracy
                       })
            raise NaN
        test_loss, test_accuracy = test(model, device, test_loader)
        best_test_accuracy = max(best_test_accuracy, test_accuracy)
        wandb.log({"Train Loss": train_loss, "Train Accuracy": train_accuracy, "Test Loss": test_loss,
                   "Test Accuracy": test_accuracy, "Best Test Accuracy": best_test_accuracy
                   })
        if test_loss > 10:
            raise NaN


def wrapped():
    run = wandb.init()
    cfg = run.config
    if not cfg["use_square"] and cfg["graft"]:
        return
    if not cfg["graft"] and (cfg["beta1"] != cfg["beta3"] or cfg["beta2"] != cfg["beta4"]):
        return
    if cfg["optimizer"] == "TGAdamW" and cfg["beta2"] != cfg["beta4"]:
        return

    try:
        run_one(**cfg)
    except NaN:
        traceback.print_exc()
        wandb.finish(1)
    wandb.finish()


def main(sweep_id: str):
    while True:
        wandb.agent(sweep_id, function=wrapped, project="truegrad-laprop", entity="clashluke")


if __name__ == '__main__':
    typer.run(main)
