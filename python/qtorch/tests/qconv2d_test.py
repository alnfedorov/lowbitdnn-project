import pytest
import torch
from qtorch.nn.functional import quantize, to_vect_c


def test_qconv2d_forward_backward(random_tensor, conv2d):
    torch.cuda.empty_cache()
    random_tensor = random_tensor.requires_grad_()
    B, C, H, W = random_tensor.shape

    fconv2d, qconv2d = conv2d
    fconv2d, qconv2d = fconv2d(C), qconv2d(C)

    if fconv2d.kernel_size[0] > H or fconv2d.kernel_size[1] > W or \
            (C % 32 == 0 and qconv2d.out_channels % 32 != 0):
        pytest.skip()
        # pytest.xfail("Convolution kernel is larger than tensor spatial size")

    # force same weights
    qconv2d.weight.data = to_vect_c(fconv2d.weight.data.clone())


    fresult = fconv2d(random_tensor)
    fresult.sum().backward() # dummy backward

    with torch.no_grad():
        fgrad_weights, fgrad_input = fconv2d.weight.grad.clone(), random_tensor.grad.clone()
        qrandom_tensor = quantize(random_tensor, to_vect_c=True, inplace=False)

    qresult = qconv2d(qrandom_tensor)
    scale = qresult.scale
    H, W = qresult.shape[2:4]
    qresult = qresult.permute([0, 1, 4, 2, 3]).reshape(-1, qconv2d.out_channels, H, W)

    assert qresult.shape == fresult.shape

    qresult = qresult.float() * scale
    qresult.sum().backward()  # dummy backward

    # delta = torch.abs(qresult - fresult)
    # # expected_eps = scale + 1e-6
    # expected_eps = 0.1
    # assert (delta <= expected_eps).all(), \
    #     f"qconv2d forward result is out of the accepted eps {expected_eps} and scale {scale}, " \
    #         f"got {delta[delta > expected_eps]} for shape {C, H, W} -> {qconv2d.out_channels, H, W}"

    with torch.no_grad:
        qgrad_weights, qgrad_input = qconv2d.weight.grad.clone(), qrandom_tensor.grad.clone()

        qgrad_weights = qgrad_weights.permute([0, 1, 4, 2, 3]).reshape(B, -1, *qgrad_weights.shape[2:4])
        assert (torch.abs(qgrad_weights - fgrad_weights) <= 1).all(), \
            "Weights gradients for qconv2d are out of the accepted eps"

        scale = qgrad_input.scale
        qgrad_input = qgrad_input.permute([0, 1, 4, 2, 3]).reshape(B, C, H, W).mul_(scale)
        assert (torch.abs(qgrad_input - fgrad_input) <= 1).all(), \
            "Input gradients for qconv2d are out of the accepted eps"