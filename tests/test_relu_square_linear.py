"""The MLP's relu(x).square() fused into the FP8 GEMM that consumes it.

The fused form keeps one FP8 tensor where the split form kept an FP8 GEMM operand
plus a bf16 relu, and recovers relu(x) for the activation's gradient by taking the
square root of that operand.

python -m pytest tests/test_relu_square_linear.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from nanochat.fp8 import Float8Linear, relu_square_linear
from nanochat.gpt import Linear

N, IN, OUT = 128, 256, 64


def unfused(x, linear):
    """What MLP.forward did before the activation and the GEMM were fused."""
    return linear(F.relu(x).square())


def make_linear(fp8, quantized=False, seed=1):
    torch.manual_seed(seed)
    linear = Linear(IN, OUT, bias=False)
    torch.nn.init.normal_(linear.weight, std=0.02)
    if fp8:
        linear = Float8Linear.from_float(linear)
        if quantized:
            linear.quantize_weight()
    return linear


def make_input(dtype=torch.float32, seed=0):
    torch.manual_seed(seed)
    # spans both sides of zero so the relu's dead half is exercised
    return (2.0 * torch.randn(N, IN, dtype=dtype)).requires_grad_(True)


def grads(out, x, linear):
    out.sum().backward()
    return x.grad.clone(), linear.weight.grad.clone()


def test_plain_linear_is_bit_identical():
    """Without FP8 there is nothing to fuse, so the result must not move at all."""
    for linear in (make_linear(fp8=False),):
        x_ref, x_got = make_input(), make_input()
        ref = unfused(x_ref, linear)
        got = relu_square_linear(x_got, linear)
        assert torch.equal(ref, got)
        ref_gx, ref_gw = grads(ref, x_ref, linear)
        linear.weight.grad = None
        got_gx, got_gw = grads(got, x_got, linear)
        assert torch.equal(ref_gx, got_gx)
        assert torch.equal(ref_gw, got_gw)


@pytest.mark.parametrize("quantized", [False, True])
def test_fp8_forward_matches_unfused(quantized):
    linear = make_linear(fp8=True, quantized=quantized)
    x_ref, x_got = make_input(), make_input()

    ref = unfused(x_ref, linear)
    got = relu_square_linear(x_got, linear)

    # identical quantization of an identical activation, so the GEMM is the same call
    assert torch.equal(ref, got)


@pytest.mark.parametrize("quantized", [False, True])
def test_fp8_gradients_match_unfused(quantized):
    linear = make_linear(fp8=True, quantized=quantized)
    x_ref, x_got = make_input(), make_input()

    ref_gx, ref_gw = grads(unfused(x_ref, linear), x_ref, linear)
    linear.weight.grad = None
    got_gx, got_gw = grads(relu_square_linear(x_got, linear), x_got, linear)

    # grad_weight is the same GEMM on the same operands
    assert torch.equal(ref_gw, got_gw)
    # grad_input differs only by relu(x) being read back through e4m3 instead of bf16
    rel = (got_gx - ref_gx).norm() / ref_gx.norm()
    assert rel < 0.02, rel  # measures 1.3%: e4m3 has 3 mantissa bits, and sqrt halves that error


def test_dead_relu_half_has_zero_gradient():
    """sqrt of the saved square must give exactly 0 where the input was negative."""
    linear = make_linear(fp8=True, quantized=True)
    x = make_input()

    relu_square_linear(x, linear).sum().backward()

    negative = x.detach() < 0
    assert negative.any()
    assert torch.equal(x.grad[negative], torch.zeros_like(x.grad[negative]))


def test_no_bf16_activation_is_retained():
    """The whole point: one FP8 [N, in] survives forward, and no bf16 copy of it."""
    linear = make_linear(fp8=True, quantized=True)
    x = make_input()

    out = relu_square_linear(x, linear)

    node = out.grad_fn
    while not hasattr(node, "saved_tensors"):  # step past the reshape back to (..., OUT)
        node = node.next_functions[0][0]
    activation_sized = [t for t in node.saved_tensors if t.numel() == N * IN]
    assert len(activation_sized) == 1, [(t.shape, t.dtype) for t in activation_sized]
    assert activation_sized[0].dtype == torch.float8_e4m3fn
