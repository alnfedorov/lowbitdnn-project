import pytest

# pytest.main()

import torch
from qtorch.nn.functional import quantize, to_vect_c, dequantize, from_vect_c
from qtorch.nn import QConv2D
from torch.nn import Conv2d
torch.manual_seed(23)
random_tensor = torch.rand((16, 1024, 5, 5), dtype=torch.float32, device="cuda", requires_grad=False).sub_(0.5).mul_(2)
random_tensor = random_tensor.requires_grad_()

qconv2d = QConv2D(1024, 1024, (3, 3), 1, 0, 1, 1, stochastic=False).to('cuda')

fconv2d = Conv2d(1024, 1024, (3, 3), 1, 0, 1, 1, bias=False).to('cuda')

B, C, H, W = random_tensor.shape

# force same weights
with torch.no_grad():
    qconv2d.weight.data = to_vect_c(fconv2d.weight.data.clone())


fresult = fconv2d(random_tensor)
fresult.sum().backward() # dummy backward

with torch.no_grad():
    fgrad_weights, fgrad_input = fconv2d.weight.grad.clone(), random_tensor.grad.clone()
    random_tensor.grad.fill_(0)

# qrandom_tensor = quantize(random_tensor, to_vect_c=True)
qresult = qconv2d(random_tensor)
# qresult = dequantize(qresult, from_vect_c=True)

assert qresult.shape == fresult.shape
qresult.sum().backward()  # dummy backward

# delta = torch.abs(qresult - fresult)
# # expected_eps = scale + 1e-6
# expected_eps = 0.1
# assert (delta <= expected_eps).all(), \
#     f"qconv2d forward result is out of the accepted eps {expected_eps} and scale {scale}, " \
#         f"got {delta[delta > expected_eps]} for shape {C, H, W} -> {qconv2d.out_channels, H, W}"

with torch.no_grad():
    qgrad_weights, qgrad_input = qconv2d.weight.grad.clone(), random_tensor.grad.clone()

    qgrad_weights = from_vect_c(qgrad_weights)

    ravel = lambda x: x.reshape(-1)
    qgrad_weights, qgrad_input = ravel(qgrad_weights), ravel(qgrad_input)
    fgrad_weights, fgrad_input = ravel(fgrad_weights), ravel(fgrad_input)

    diff = torch.abs(qgrad_weights - fgrad_weights)
    ind = torch.argmax(diff)
    print(f"Weight gradients: max abs diff {diff[ind]}, max rel diff {diff[ind]/fgrad_weights[ind]}, mean {diff.mean()}",)
    # assert (torch.abs(qgrad_weights - fgrad_weights) <= 1).all(), \
    #     "Weights gradients for qconv2d are out of the accepted eps"


    diff = torch.abs(qgrad_input - fgrad_input)
    ind = torch.argmax(diff)
    print(f"Input gradients: max abs diff {diff[ind]}, max rel diff {diff[ind]/fgrad_input[ind]}, mean {diff.mean()}",)
    # assert (torch.abs(qgrad_input - fgrad_input) <= 1).all(), \
    #     "Input gradients for qconv2d are out of the accepted eps"
# 46.9 <-> 0.1290
pass
# def _compare_qconv(input, iscale, weights, wscale, stride, padding, dilation, groups):
#     B, C, H, W = input.shape
#     num_filters, _, filter_height, filter_width = weights.shape
#     assert C % 32==0 or C % 4==0
#     if C % 32 == 0:
#         tmp_input = input.reshape(B, C // 32, 32, H, W)
#         tmp_weights = weights.reshape(num_filters, C // 32, 32, filter_height, filter_width)
#     else:
#         tmp_input = input.reshape(B, C // 4, 4, H, W)
#         tmp_weights = weights.reshape(num_filters, C // 4, 4, filter_height, filter_width)
#
#     tmp_input, tmp_weights = tmp_input.permute([0, 1, 3, 4, 2]), tmp_weights.permute([0, 1, 3, 4, 2])
#     rscale, qresult = qconv2d(tmp_input, iscale, tmp_weights, wscale, stride, padding, dilation, groups)
#     B, C, H, W = qresult.shape[0], qresult.shape[1] * qresult.shape[-1], qresult.shape[2], qresult.shape[3]
#     qresult = qresult.permute([0, 1, 4, 2, 3]).reshape(B, C, H, W).contiguous()
#
#     rscale_, qconv_scale = _compute_result_scale(iscale, wscale)
#     assert abs(rscale - rscale_) <= 1e-9
#
#     # Extremely close to what cudnn does and what we expect
#     fresult = F.conv2d(input.float(), weights.float(), None, stride, padding, dilation, groups)
#     fresult *= qconv_scale
#     fresult = fresult.round_().clamp_(-128, 127).to(torch.int8)
#
#     delta = torch.abs(qresult.to(torch.int32) - fresult.to(torch.int32))
#     assert (delta <= 1).all()
#
#
# def _compare_qpool2d(input, kernel, stride, padding):
#     B, C, H, W = input.shape
#     assert C % 32 == 0 or C % 4 == 0
#     if C % 32 == 0:
#         tmp_input = input.reshape(B, C // 32, 32, H, W)
#     else:
#         tmp_input = input.reshape(B, C // 4, 4, H, W)
#     tmp_input = tmp_input.permute([0, 1, 3, 4, 2])
#
#     qresult = qmax_pool2d(tmp_input, kernel, stride, padding)
#     B, C, H, W = qresult.shape[0], qresult.shape[1] * qresult.shape[-1], qresult.shape[2], qresult.shape[3]
#     qresult = qresult.permute([0, 1, 4, 2, 3]).reshape(B, C, H, W).contiguous()
#
#     fresult = F.max_pool2d(input.float(), kernel, stride, padding)
#     fresult = fresult.round_().clamp_(-128, 127).to(torch.int8)
#
#     delta = torch.abs(qresult.to(torch.int32) - fresult.to(torch.int32))
#     assert (delta <= 0).all()
#
#
# def _test_qconv2d(batch, dimensions, channels, filters, input_values, weight_values, input_scales, weight_scales):
#     # Other values are simply not supported or not relevant yet
#     stride, padding, dilation, groups = (1, 1), (0, 0), (1, 1), 1
#     for B in batch:
#         for H, W in dimensions:
#             for C in channels:
#                 input_shape = (B, C, H, W)
#                 input = torch.ones(input_shape, dtype=torch.int8, requires_grad=False, device='cuda')
#                 for convolutions in filters:
#                     if convolutions < C:
#                         continue
#                     print(f"Run conv test with shape {input_shape} -> (3, 3) -> {(B, convolutions, H, W)}")
#                     weight_shape = (convolutions, C, 3, 3) # TODO: Other kernel shapes
#                     weights = torch.ones(weight_shape, dtype=torch.int8, requires_grad=False, device='cuda')
#
#
#                     for ivalue, wvalue in zip(input_values, weight_values):
#                         input.fill_(ivalue)
#                         weights.fill_(wvalue)
#                         for iscale, wscale in zip(input_scales, weight_scales):
#                             _compare_qconv(input.clone(), iscale, weights.clone(),
#                                            wscale, stride, padding, dilation, groups)
#                             gc.collect()
#                             torch.cuda.empty_cache()
#
#
# def _test_qpool2d(batch, dimensions, channels, input_values):
#     # Other values are simply not supported or not relevant yet
#     kernel, stride, padding = (2, 2), (1, 1), (0, 0)
#     for B in batch:
#         for H, W in dimensions:
#             for C in channels:
#                 input_shape = (B, C, H, W)
#                 print(f"Run max_pool test with shape {input_shape} -> {(B, C, H//2, W//2)}")
#                 input = torch.ones(input_shape, dtype=torch.int8, requires_grad=False, device='cuda')
#                 for ivalue in input_values:
#                     input.fill_(ivalue)
#                     _compare_qpool2d(input.clone(), kernel, stride, padding)
#                     gc.collect()
#                     torch.cuda.empty_cache()
#
#
# def _test_qconv2d_random(batch, dimensions, channels, filters, input_scales, weight_scales):
#     # Other values are simply not supported or not relevant yet
#     stride, padding, dilation, groups = (1, 1), (0, 0), (1, 1), 1
#     for B in batch:
#         for H, W in dimensions:
#             for C in channels:
#                 for convolutions in filters:
#                     if convolutions < C:
#                         continue
#                     input_shape = (B, C, H, W)
#                     input = torch.randint(-1, 1, input_shape, dtype=torch.int8, requires_grad=False, device='cuda')
#
#                     print(f"Run random conv test with shape {input_shape} -> {(B, convolutions, H, W)}")
#                     weight_shape = (convolutions, C, 3, 3) # TODO: Other kernel shapes
#                     weights = torch.randint(-1, 1, weight_shape, dtype=torch.int8, requires_grad=False, device='cuda')
#                     for iscale, wscale in zip(input_scales, weight_scales):
#                         _compare_qconv(input.clone(), iscale, weights.clone(), wscale, stride, padding, dilation, groups)
#                         gc.collect()
#                         torch.cuda.empty_cache()
#
#
# def _test_qpool2d_random(batch, dimensions, channels):
#     # Other values are simply not supported or not relevant yet
#     kernel, stride, padding = (2, 2), (2, 2), (0, 0)
#     for B in batch:
#         for H, W in dimensions:
#             for C in channels:
#                 input_shape = (B, C, H, W)
#                 print(f"Run random max_pool test with shape {input_shape} -> {(B, C, H//2, W//2)}")
#                 input = torch.randint(-1, 1, input_shape, dtype=torch.int8, requires_grad=False, device='cuda')
#                 _compare_qpool2d(input, kernel, stride, padding)
#                 gc.collect()
#                 torch.cuda.empty_cache()
#
#
# def test_qconv2d_simple():
#     batch = (8, 16, 32, 64,)
#     H_W = [(3, 3), (7, 7), (32, 32)]
#     channels = (4, 8, 16, 32, 64)
#     filters = (4, 8, 16, 32, 64)
#
#     input_values, weight_values = [x.ravel() for x in np.meshgrid([127, -128, 3, -2, 0],
#                                                                   [127, -2, 3, 0 -128])]
#     input_scales, weight_scales = [x.ravel() for x in np.meshgrid([0.3, 2.5, 1.0],
#                                                                   [0.1, 1.0, 2.3])]
#     # _test_qconv2d(batch, H_W, channels, filters, input_values, weight_values, input_scales, weight_scales)
#     _test_qconv2d_random(batch, H_W, channels, filters, input_scales, weight_scales)
#
#
# def test_qpool2d_simple():
#     batch = (8, 16, 32)
#     H_W = [(4, 4), (15, 15), (32, 32)]
#     channels = (4, 8, 16, 32, 64)
#     input_values = [127, -128, 0, 1, 2, -3]
#     # _test_qpool2d(batch, H_W, channels, input_values)
#     _test_qpool2d_random(batch, H_W, channels)
