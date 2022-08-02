import math
import typing

import torch


class ActivationFn(torch.autograd.Function):
    @staticmethod
    def core(x: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        raise NotImplementedError

    @staticmethod
    def grad(x: torch.Tensor, *intermediates: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def native(cls, x: torch.Tensor) -> torch.Tensor:
        return cls.core(x)[0]

    @classmethod
    def forward(cls, ctx, x):
        out, intermediates = cls.core(x)
        if x.requires_grad:
            ctx.save_for_backward(cls.grad(x, *intermediates))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 0:
            return None
        grad, = ctx.saved_tensors
        return grad_output * grad


class MishFn(ActivationFn):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681
    """

    @staticmethod
    def core(x: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        x_tanh_sp = torch.nn.functional.softplus(x).tanh()
        return x * x_tanh_sp, [x_tanh_sp]

    @staticmethod
    def grad(x: torch.Tensor, x_tanh_sp: torch.Tensor) -> torch.Tensor:
        return x_tanh_sp + x * x.sigmoid() * (1 - x_tanh_sp.square())

    @staticmethod
    def native(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mish(x)


class SwishFn(ActivationFn):
    """
    Gaussian Error Linear Units (GELUs)
    https://arxiv.org/abs/1606.08415
    """

    @staticmethod
    def core(x: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        sigmoid_x = x.sigmoid()
        swish_x = x * sigmoid_x
        return swish_x, [sigmoid_x, swish_x]

    @staticmethod
    def grad(x: torch.Tensor, sigmoid_x: torch.Tensor, swish_x: torch.Tensor) -> torch.Tensor:
        return sigmoid_x + swish_x * (1 - sigmoid_x)

    @staticmethod
    def native(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(x)


class LeCunTanhFn(ActivationFn):
    @staticmethod
    def core(x: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        tanh_x = x.tanh()
        return tanh_x + 0.1 * x, [tanh_x]

    @staticmethod
    def grad(x: torch.Tensor, tanh_x: torch.Tensor) -> torch.Tensor:
        return 1.1 - tanh_x ** 2


class GeLUFn(ActivationFn):
    @staticmethod
    def core(x: torch.Tensor) -> typing.Tuple[torch.Tensor, typing.List[torch.Tensor]]:
        x3 = x ** 3
        xpi = x * (2 / math.pi) ** 0.5
        mid = 0.044715 * x3 + xpi
        t = mid.tanh()
        t1 = t + 1
        return x * t1 * 0.5, [x3, xpi, mid, t, t1]

    @staticmethod
    def grad(x: torch.Tensor, x3: torch.Tensor, xpi: torch.Tensor, mid: torch.Tensor, t: torch.Tensor,
             t1: torch.Tensor) -> torch.Tensor:
        return 0.5 * (t1 + (1 - t ** 2) * (0.134145 * x3 + xpi))

    @staticmethod
    def native(x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.gelu(x)


class Activation(torch.nn.Module):
    activation: ActivationFn

    def __init__(self, native: bool = False):
        super(Activation, self).__init__()
        self.forward = self.activation.native if native else self.activation.apply


class Mish(Activation):
    activation = MishFn


class Swish(Activation):
    activation = SwishFn


class LeCunTanh(Activation):
    activation = LeCunTanhFn


class GeLU(Activation):
    activation = GeLUFn
