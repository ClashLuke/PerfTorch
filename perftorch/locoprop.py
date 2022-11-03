from typing import Callable, Optional

import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module, functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LocopropFn(Function):
    @staticmethod
    def forward(ctx, inp: Tensor, mod: Module, optim: type, base_lr: float, act_lr: float, update_iterations: int, ):
        ctx.mod = mod
        ctx.optim = optim
        ctx.base_lr = base_lr
        ctx.act_lr = act_lr
        ctx.update_iterations = update_iterations
        ctx.save_for_backward(inp.detach())
        return mod(inp).requires_grad_(inp.requires_grad)

    @staticmethod
    def backward(ctx, dy: Tensor):
        inp, = ctx.saved_tensors
        optim: Optimizer = ctx.optim(ctx.mod.parameters(), ctx.base_lr)
        sched = LambdaLR(optim, lambda step: (step + 8) ** -1)
        target = None
        inp = inp.requires_grad_(True)
        for p in ctx.mod.parameters():
            p.old_data = p.data.clone()
        for i in range(ctx.update_iterations):
            inp.grad = None
            with torch.enable_grad():
                # inp.requires_grad_(True)
                out = ctx.mod(inp)
                if i == 0:
                    target = (out - ctx.act_lr * dy).detach()
                (out - target).square().div(out.numel() / out.size(0)).sum().backward()
                if i == 0:
                    dx = inp.grad.detach().requires_grad_(True)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
        # dx = inp.grad.detach().requires_grad_(True)
        inp.grad = None
        for p in ctx.mod.parameters():
            p.grad = p.old_data - p.data
            p.data = p.old_data
            p.old_data = None
        return dx, None, None, None, None, None, None


class LocopropWrapper(Module):
    def __init__(self, module: Module, base_lr: float, act_lr: float = 1, update_iterations: int = 10,
                 optim_cls: type = torch.optim.SGD):
        super(LocopropWrapper, self).__init__()
        self.inner_module = module
        self.base_lr = base_lr
        self.act_lr = act_lr
        self.update_iterations = update_iterations
        self.optim_cls = optim_cls
        self.step = 0

    def forward(self, inp: Tensor):
        inp.requires_grad_(self.training)
        return LocopropFn.apply(inp, self.inner_module, self.optim_cls, self.base_lr, self.act_lr,
                                self.update_iterations)


class ResLocoprop(Module):
    def __init__(self, module: Module):
        super(ResLocoprop, self).__init__()
        self.inner_module = module
        self.original: Optional[Tensor] = None
        self.target: Optional[Tensor] = None
        self.retain_grad = True

    def forward(self, inp: Tensor):
        if self.training and self.retain_grad:
            self.input = inp
        out = self.inner_module(inp)
        if self.training:
            if self.retain_grad:
                out.retain_grad()
            self.original = out
        return out


def old_locoprop_step(net: Module, closure: Callable[[], torch.Tensor], steps: int, inner_lr: float,
                      inner_optimizer: Optimizer) -> torch.Tensor:
    # Get target hidden states
    net.requires_grad_(False)
    loss = closure()
    loss.backward()
    inner_optimizer.zero_grad()

    # Optimize hidden states
    net.requires_grad_(True)
    original_weights = list(torch.clone(p) for p in net.parameters())
    for m in net.modules():
        if not isinstance(m, ResLocoprop):
            continue
        with torch.no_grad():
            target = torch.clone(m.original.detach() - m.original.grad * inner_lr).detach().requires_grad_(True)
        inp = torch.clone(m.input.detach()).requires_grad_(True)
        m.retain_grad = False
        for i in range(steps):
            m.zero_grad()
            with torch.enable_grad():
                loss = (m(inp) - target).square().mean()
                loss.backward()
                if i == 0:
                    first_loss = loss.item()
            inner_optimizer.step()
        m.retain_grad = True
        print(loss.item(), first_loss, first_loss > loss.item())
    print('\n------\n')
    with torch.no_grad():
        for o, p in zip(original_weights, net.parameters()):
            p.grad = o.data - p.data
            p.data = o.data
    net.requires_grad_(True)

    return loss


def locoprop_step(net: Module, closure: Callable[[], torch.Tensor], steps: int, inner_lr: float,
                  inner_optimizer: type, lr: float) -> torch.Tensor:
    inner_lr = 1
    # Get target hidden states
    net.requires_grad_(False)
    loss = closure()
    loss.backward()

    # Optimize hidden states
    net.requires_grad_(True)
    original_weights = list(torch.clone(p) for p in net.parameters())
    looped = []
    with torch.no_grad():
        for m in net.modules():
            if not isinstance(m, ResLocoprop):
                continue
            target = torch.clone(m.original.detach() - m.original.grad * inner_lr).detach().requires_grad_(True)
            inp = torch.clone(m.input.detach()).requires_grad_(True)
            m.retain_grad = False
            looped.append((m, target, inp))
    inner_optimizer = inner_optimizer([p for m, _, _ in looped for p in m.parameters()], lr=lr)
    for i in range(steps):
        losses = 0
        for m, tgt, inp in looped:
            with torch.enable_grad():
                losses += (m(inp) - tgt).square().mean()
        losses.backward()
        inner_optimizer.step()
        inner_optimizer.zero_grad()
        if i == 0:
            first_loss = loss.item()
    for m, _, _ in looped:
        m.retain_grad = True
    print(loss.item(), first_loss, first_loss > loss.item())
    print('------')
    with torch.no_grad():
        for o, p in zip(original_weights, net.parameters()):
            p.grad = o.data - p.data
            p.data = o.data
    net.requires_grad_(True)

    return loss


class ConvergenceError(ValueError):
    pass


class LocoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, mod: torch.nn.Module, iterations, lr):
        ctx.mod = mod
        ctx.inp = inp
        ctx.iterations = iterations
        ctx.lr = lr
        return mod(inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        mod: torch.nn.Module = ctx.mod
        inp: torch.Tensor = ctx.inp.detach()
        opt = torch.optim.SGD(mod.parameters(), lr=ctx.lr)
        with torch.no_grad():
            out = (mod(inp).detach() - dy).detach()
        grad = None
        mod.step += 1
        with torch.enable_grad():
            inp.requires_grad_(True)
            for i in range(ctx.iterations):
                loss = F.mse_loss(mod(inp), out, reduction="mean")
                loss.backward()
                opt.step()
                mod.zero_grad()
                if i == 0:
                    first_loss = loss.item()
                    grad = inp.grad * dy.numel()
                    inp.requires_grad_(False)
                    inp = inp.detach()
        if first_loss < loss.item():
            raise ConvergenceError(f"Error larger after step {mod.step}")
        return grad, None, None, None


class Loco(torch.nn.Module):
    def __init__(self, mod: torch.nn.Module, iterations: int, lr: float):
        super(Loco, self).__init__()
        self.mod = mod
        self.iterations = iterations
        self.lr = lr
        mod.step = 0

    def forward(self, inp: torch.Tensor):
        return LocoFn.apply(inp, self.mod, self.iterations, self.lr)


class NewGradFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, mod: torch.nn.Module, iterations, lr):
        ctx.mod = mod
        ctx.inp = inp
        ctx.iterations = iterations
        ctx.lr = lr
        return mod(inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        mod: torch.nn.Module = ctx.mod
        inp: torch.Tensor = ctx.inp.detach()
        opt = torch.optim.SGD(mod.parameters(), lr=ctx.lr)
        with torch.enable_grad():
            if ctx.iterations == 0:
                inp.requires_grad_(True)
            (mod(inp) * dy).sum().backward()
        opt.step()
        if ctx.iterations == 0:
            return inp.grad, None, None, None

        mod.requires_grad_(False)
        with torch.enable_grad():
            inp.requires_grad_(True)
            inp.retain_grad()
            (mod(inp) * dy).sum().backward()
            grad = inp.grad
        mod.requires_grad_(True)
        mod.zero_grad()
        return grad, None, None, None


class NewGrad(torch.nn.Module):
    def __init__(self, mod: torch.nn.Module, iterations: int, lr: float):
        super(NewGrad, self).__init__()
        self.mod = mod
        self.lr = lr
        self.iterations = iterations

    def forward(self, inp: torch.Tensor):
        return NewGradFn.apply(inp, self.mod, self.iterations, self.lr)


class LocoFn2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, mod: torch.nn.Sequential, iterations, lr):
        ctx.mod = mod
        ctx.inp = inp
        ctx.iterations = iterations
        ctx.lr = lr
        return mod(inp)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        mod: torch.nn.Sequential = ctx.mod
        mod.step += 1

        inp: torch.Tensor = ctx.inp.detach()
        inp.requires_grad_(True)
        opt = torch.optim.SGD(mod.parameters(), lr=ctx.lr)
        out = inp
        outputs = []
        if ctx.iterations > 0:
            mod.requires_grad_(False)
        with torch.enable_grad():
            for m in mod:
                out = m(out)
                out.requires_grad_(True)
                out.retain_grad()
                outputs.append(out)
            F.mse_loss(out, (out - dy).detach(), reduction="sum").backward()
            grad = inp.grad
        if ctx.iterations == 0:
            opt.step()
            mod.zero_grad()
            return grad, None, None, None

        mod.zero_grad()
        outputs = [(out - out.grad).detach() for out in outputs]
        inp = inp.detach()
        mod.requires_grad_(True)

        with torch.enable_grad():
            for i in range(ctx.iterations):
                out = inp.detach()
                out.requires_grad_(True)
                losses = []
                for m, tgt in zip(mod, outputs):
                    out = m(out)
                    losses.append(F.mse_loss(out, tgt, reduction="sum"))
                loss = sum(losses)
                loss.backward()
                for depth, m in enumerate(mod):
                    for p in m.parameters():
                        p.grad /= len(mod) - depth
                opt.step()
                mod.zero_grad()
                if i == 0:
                    first_loss = loss.item()
        if first_loss < loss.item():
            raise ConvergenceError(f"Error larger after step {mod.step}. Before: {first_loss}, after {loss.item()}")
        return grad, None, None, None


class Loco2(torch.nn.Module):
    def __init__(self, mod: torch.nn.Sequential, iterations: int, lr: float):
        super(Loco2, self).__init__()
        self.mod = mod
        self.iterations = iterations
        self.lr = lr
        mod.step = 0

    def forward(self, inp: torch.Tensor):
        if self.training:
            inp = inp.requires_grad_(True)
        return LocoFn2.apply(inp, self.mod, self.iterations, self.lr)
