import math

import torch


class LocoFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.Tensor, mod: torch.nn.Sequential, locoprop_step, iterations, lr):
        ctx.mod = mod
        ctx.inp = inp
        ctx.iterations = iterations
        ctx.locoprop_step = locoprop_step
        ctx.lr = lr
        return mod(inp)

    @staticmethod
    def _step(ctx, dy: torch.Tensor):
        mod: torch.nn.Sequential = ctx.mod
        mod.step += 1

        inp: torch.Tensor = ctx.inp.detach()
        inp.requires_grad_(True)
        out = inp
        outputs = []
        if ctx.iterations > 0:
            mod.requires_grad_(False)
        else:
            mod.requires_grad_(True)
        with torch.enable_grad():
            for m in mod:
                out = m(out)
                out.requires_grad_(True)
                out.retain_grad()
                outputs.append(out)
            torch.autograd.backward(out, dy.detach())
            input_gradient = torch.clone(inp.grad.detach().contiguous())
        if ctx.iterations == 0:
            for p in mod.parameters():
                p.grad.data /= inp.size(0)
            torch.optim.SGD(mod.parameters(), lr=ctx.lr).step()
            mod.zero_grad()
            return input_gradient

        mod.zero_grad()
        outputs = [(out - out.grad * out.size(0)).detach() for out in outputs]
        out = inp - input_gradient
        mod.requires_grad_(True)

        with torch.enable_grad():
            for m, tgt in zip(mod, outputs):
                params = list(m.parameters())
                if not params:
                    out = m(out)
                    continue
                m_inp = out.detach().requires_grad_(True)
                last_error = 1e9
                opt = torch.optim.SGD(m.parameters(), lr=ctx.lr)
                prev_params = [torch.clone(p) for p in params]
                for i in range(4000):
                    out = m(m_inp).contiguous()
                    grd = (out - tgt).contiguous().detach()
                    error = grd.norm().item()
                    # if i % 100 == 0:
                    #     print(f'Iteration: {i} - LR: {int(math.log2(opt.param_groups[0]["lr"]))} - Last: {last_error} - Current: {error}')
                    if i > 0:
                        if error > last_error:
                            for param_group in opt.param_groups:
                                param_group["lr"] /= 2
                                for cp, pp in zip(params, prev_params):
                                    cp.data = pp.data
                            if any(g["lr"] < 1e-6 for g in opt.param_groups):
                                break
                            continue
                        # elif error < last_error:
                        #     for param_group in opt.param_groups:
                        #         param_group["lr"] *= 2
                    if last_error - error < 1e-4:
                        break
                    last_error = error
                    prev_params = [torch.clone(p) for p in params]
                    torch.autograd.backward(out, grd / tgt.numel() * out.size(0), inputs=params)
                    opt.step()
                    m.zero_grad()
                if error > last_error:
                    for cp, pp in zip(params, prev_params):
                        cp.data = pp.data
                # print('\n')
            # print('\n\n\n')
        return input_gradient

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        original_weights = [torch.clone(p.data.detach()) for p in ctx.mod.parameters()]
        grad = LocoFn._step(ctx, dy)
        for op, p in zip(original_weights, ctx.mod.parameters()):
            p.grad = (op - p).detach()
            p.data = op.data
        return grad, None, None, None, None


class Loco(torch.nn.Module):
    def __init__(self, mod: torch.nn.Sequential, locoprop_step: float, iterations: int, lr: float):
        super(Loco, self).__init__()
        self.mod = mod
        self.locoprop_step = locoprop_step
        self.iterations = iterations
        self.lr = lr
        mod.step = 0

    def forward(self, inp: torch.Tensor):
        if self.training:
            inp = inp.requires_grad_(True)
        return LocoFn.apply(inp, self.mod, self.locoprop_step, self.iterations, self.lr)
