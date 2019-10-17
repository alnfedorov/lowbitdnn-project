import torch
from torch import nn
from qtorch.nn.functional import qconv2d, to_vect_c


class QConv2D(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, num_bits=8, min_value=None, max_value=None, stochastic=True):
        super(QConv2D, self).__init__(in_channels, out_channels, kernel_size, stride,
                                      padding, dilation, groups, False)
        self.num_bits = num_bits
        self.min_value, self.max_value = min_value, max_value
        self.stochastic = stochastic
        self.weight.data = to_vect_c(self.weight.data)

    def forward(self, input):
        # stochastic = True if self.training and self.stochastic else False

        results = qconv2d(input, self.weight, self.stride,
                          self.padding, self.dilation, self.groups)

        return results

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     res = super().state_dict(destination, prefix, keep_vars)
    #     key = prefix + "weight"
    #     res[key] = quantize(res[key], self.num_bits, self.min_value, self.max_value)
    #     return res
    #
    # def load_state_dict(self, state_dict, *args, **kwargs):
    #     state_dict['weight'] = dequantize(state_dict['weight'], from_vect_c=False)
    #     super().load_state_dict(state_dict, *args, **kwargs)
