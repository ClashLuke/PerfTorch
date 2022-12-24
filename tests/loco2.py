import collections
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import wandb
from perftorch.locoprop import Loco


class ConvBlock(nn.Sequential):
    def __init__(self, features: int, stride: int):
        super().__init__()
        self.add_module("relu", nn.ReLU())
        # self.add_module("norm", LayerNorm2d(features))
        self.add_module("input", nn.Linear(features, features, bias=False))


class GlobalAvgPool(nn.Module):
    def forward(self, inp: torch.Tensor):
        return inp.mean((2, 3))


class Head(nn.Sequential):
    def __init__(self, feature_factor: int):
        super(Head, self).__init__()
        self.add_module("classifier", nn.Linear(feature_factor, 10))


class Net(nn.Sequential):
    def __init__(self, feature_factor: int):
        super(Net, self).__init__()
        self.add_module("input", nn.Conv2d(1, feature_factor, 3, 2))
        self.add_module("mean_pool", GlobalAvgPool())
        for i in range(6):
            self.add_module(f"stage{i}", Loco(ConvBlock(feature_factor, 1 + int(i > 0 and i % 2)), 1, 8, 1))
            self.add_module(f"norm{i}", nn.LayerNorm(feature_factor))
        self.add_module("classifier", Head(feature_factor))


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
        global_loss += loss.detach() * target.size(0)
        loss.backward()
        optimizer.step()
    global_loss /= len(train_loader.dataset)
    accuracy = accuracy.float()
    accuracy /= len(train_loader.dataset)
    return global_loss.item(), accuracy.item()


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


def get_model(feature_factor: int, device: typing.Union[str, torch.device], locoprop_step, iterations, lr) -> Net:
    return Net(feature_factor).to(device)


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
def run_one(seed: int, feature_factor: int, batch_size: int, epochs: int, locoprop_step: float,
            locoprop_iter: int, locoprop_lr: float, learning_rate: float, outer_optimizer: type
            ) -> typing.Iterable[typing.Tuple[float, float]]:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.empty_cache()

    torch.manual_seed(seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    model = get_model(feature_factor, device, locoprop_step, locoprop_iter, locoprop_lr)
    optimizer: torch.optim.Optimizer = outer_optimizer(model.parameters(), lr=learning_rate)

    train_loader = get_dataset(batch_size, True)
    test_loader = get_dataset(16_384, False)

    for _ in range(epochs):
        tr_loss, tr_acc = train(model, device, train_loader, optimizer)
        te_loss, te_acc = test(model, device, test_loader)
        yield tr_loss, tr_acc, te_loss, te_acc
    if use_cuda:
        torch.cuda.empty_cache()


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


def main():
    options = {"locoprop_iter": [0],
               "learning_rate": [10 ** -i for i in range(1, 5)],
               "feature_factor": [16],
               "locoprop_lr": [10 ** -i for i in range(1)],
               "locoprop_step": [10 ** -i for i in range(31)],
               "batch_size": [1],
               "outer_optimizer": [torch.optim.SGD, torch.optim.AdamW, torch.optim.RMSprop],
               "seed": [0, 1],
               }
    configs = list(product(options))

    for conf in tqdm.tqdm(configs):
        wandb.init(project="locoprop-2", entity="clashluke", reinit=True, config=conf)
        best_te_acc = 0
        for tr_loss, tr_acc, te_loss, te_acc in run_one(epochs=8, **conf):
            best_te_acc = max(best_te_acc, te_acc)
            print(tr_loss, tr_acc, te_loss, te_acc)
            if tr_loss > 10 or tr_loss != tr_loss or te_loss > 10 or te_loss != te_loss:
                break
            wandb.log({"Train Loss": tr_loss, "Train Accuracy": tr_acc, "Test Loss": te_loss, "Test Accuracy": te_acc,
                       "Peak Test Accuracy": best_te_acc
                       })
        wandb.finish()


if __name__ == '__main__':
    main()
