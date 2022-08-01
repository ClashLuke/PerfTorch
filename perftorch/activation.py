import torch

class MishFn(torch.autograd.Function):
    """
    Mish: A Self Regularized Non-Monotonic Neural Activation Function
    https://arxiv.org/abs/1908.08681
    """
    @staticmethod
    def forward(ctx, x):
        x_tanh_sp = F.softplus(x).tanh()
        if x.requires_grad:
            ctx.save_for_backward(x_tanh_sp + x * x.sigmoid() * (1 - x_tanh_sp.square()))
        y = x * x_tanh_sp
        return y

    @staticmethod
    def backward(ctx, grad_output):
        if len(ctx.saved_tensors) == 0:
            return None
        grad, = ctx.saved_tensors
        return grad_output * grad
        
        
class SwishFn(torch.autograd.Function):
    """
    Gaussian Error Linear Units (GELUs) 
    https://arxiv.org/abs/1606.08415
    """
    @staticmethod
    def forward(ctx, i):
        sigmoid_i = i.sigmoid()
        result = i * sigmoid_i
        if i.requires_grad:
            ctx.save_for_backward(sigmoid_i + result * (1 - sigmoid_i))
        return result

    @staticmethod
    def backward(ctx, grad_output):
        grad, = ctx.saved_tensors
        return grad_output * grad
