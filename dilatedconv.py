import torch


@torch.jit.script
def flt(x: torch.Tensor, n):
    return torch.logical_or(x >= n, x < 0)


class Unfold(torch.jit.ScriptModule):
    def __init__(self, groups, features, size):
        super(Unfold, self).__init__()
        feat = features
        features //= groups
        with torch.no_grad():
            mask_kernel = torch.arange(-1, 2).view(-1, 1)
            pre_features = torch.cat([torch.tensor([2 ** i] * features) for i in range(groups)]).view(-1, 1, 1, 1, 1)
            pre_kernel = mask_kernel.add(3 * mask_kernel.view(1, -1)).view(1, 3, 3, 1, 1) * pre_features
            mask_kernel = mask_kernel.view(1, 1, -1) * pre_features.view(-1, 1, 1)
            theight = torch.tensor([size])
            hkernel = torch.arange(0, size).view(-1, 1).view(1, 1, 1, 1, -1)
            wkernel = torch.arange(0, size * size, size).view(-1, 1).view(1, 1, 1, -1, 1)
            fkernel = torch.arange(0, size * size * feat, size * size).view(-1, 1, 1, 1, 1)
            kernel = fkernel.add(wkernel).add(hkernel).add(pre_kernel).view(1, -1)
            height_mask = mask_kernel.mul(3).add(hkernel.view(1, -1, 1))
            mask = torch.logical_or(flt(height_mask.add(mask_kernel * 2).view(feat, -1, 1), theight),
                                    flt(mask_kernel.mul(3).add(hkernel.view(1, -1, 1)
                                                               ).view(feat, -1, 1).transpose(1, 2),
                                        theight)
                                    )
            mask = mask.view(feat, size, 3, size, 3).transpose(1, -1).transpose(-2, -1).transpose(1, 2).reshape(1, -1)
            kernel = torch.where(mask, torch.zeros(1, dtype=torch.long).expand_as(mask), kernel + 1)
        self.register_buffer('kernel', kernel)

    def forward(self, inp):
        batch, feat, width, height = inp.size()
        output = torch.cat([torch.zeros((batch, 1), device=inp.device, dtype=inp.dtype), inp.view(batch, -1)], 1)
        output = output.gather(1, self.kernel.expand(batch, -1))
        return output.view(batch, feat * 9, width * height)


class DilatedConv(torch.jit.ScriptModule):
    __constants__ = ['out_features']

    def __init__(self, in_features, out_features, groups, size, kernel_size=3):
        super().__init__()
        if in_features % groups != 0:
            raise UserWarning("Nope")
        self.unfold = Unfold(groups, in_features, size)
        self.out_features = out_features
        self.weight = torch.nn.Parameter(torch.empty(out_features, kernel_size ** 2 * in_features))
        torch.nn.init.orthogonal_(self.weight)

    def forward(self, inp):
        size = list(inp.size())
        size[1] = self.out_features
        data = self.unfold(inp)
        data = self.weight.unsqueeze(0).expand(size[0], -1, -1).bmm(data)
        data = data.view(*size)
        return data


class NaiveDilatedConv(torch.jit.ScriptModule):
    def __init__(self, in_features, out_features, groups):
        super().__init__()
        self.in_features = in_features // groups
        self.convs = torch.nn.ModuleList([torch.nn.Conv2d(in_features // groups, out_features, 3,
                                                          dilation=2 ** i, padding=2 ** i)
                                          for i in range(groups)])

    def forward(self, inp):
        out = self.convs[0](inp[:, :self.in_features])
        for i, conv in enumerate(self.convs[1:], 1):
            out = out + conv(inp[:, self.in_features * i:self.in_features * (1 + i)])
        return out
