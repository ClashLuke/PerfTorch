import torch
from torch import Tensor
from torch.autograd import Function
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class LocopropFn(Function):
    @staticmethod
    def forward(ctx, inp: Tensor, mod: Module, optim: type, base_lr: float, act_lr: float, update_iterations: int,):
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
        sched = LambdaLR(optim, lambda step: (step + 1) ** -2)
        dx = None
        target = None
        inp = inp.requires_grad_(True)
        for p in ctx.mod.parameters():
            p.old_data = p.data.clone()
        for i in range(ctx.update_iterations):
            with torch.enable_grad():
                out = ctx.mod(inp)
                if i == 0:
                    target = (out - ctx.act_lr * dy).detach()
                out = (out - target).square()
                if i == 0:
                    out.sum().backward()
                    dx = inp.grad.detach().requires_grad_(True)
                else:
                    out.mean().backward()
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=False)
            inp.grad = None
        for p in ctx.mod.parameters():
            p.grad = p.old_data - p.data
            p.data = p.old_data
            p.old_data = None
        return dx, None, None, None, None, None, None


class LocopropWrapper(Module):
    def __init__(self, module: Module, base_lr: float, act_lr: float = 1, update_iterations: int = 10, optim_cls: type = torch.optim.SGD):
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
