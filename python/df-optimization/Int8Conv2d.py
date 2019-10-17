import torch
from torch.nn import Module, Parameter
from qtorch.nn.functional import qconv2d


class Int8Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation):
        super(Int8Conv2d, self).__init__()
        if in_channels % 32 == 0 and out_channels % 32 == 0:
            self.vect_c = 4
        elif in_channels % 4 == 0 and out_channels % 4 == 0:
            self.vect_c = 4
        else:
            raise NotImplementedError()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.scale = 1  # TODO fix to be arbitrary

        self.weight = Parameter(torch.Tensor(out_channels, in_channels // self.vect_c, *kernel_size, self.vect_c).to(torch.int8),
                                requires_grad=False)
        self.reset_parameters()

    def reset_parameters(self):
        fill = torch.randint_like(self.weight, -5, 5, dtype=torch.int8, requires_grad=False)
        torch.s_copy_(self.weight, fill)

    def forward(self, scale, tensor):
        assert tensor.dim() == 5 and tensor.size(4) == self.vect_c \
               and tensor.size(1) == self.in_channels // self.vect_c and tensor.dtype == torch.int8
        return qconv2d(tensor, scale, self.weight, self.scale, self.stride, self.padding, self.dilation, 1)
