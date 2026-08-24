"""
Test the FP8 quantization path: the compiled weight-quantization used once per
optimizer step must produce byte-identical results to quantizing per call.

python -m pytest tests/test_fp8.py -v
"""

import pytest
import torch

from nanochat.fp8 import _to_fp8, _to_fp8_compiled

# One narrow and one wide, both with dims divisible by 16 as the FP8 gate requires.
SHAPES = [(64, 128), (1536, 1536), (2048, 1536)]


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compiled_quantization_is_byte_identical_to_eager(shape, dtype):
    """Compiling the quantization must not change a single bit of its output."""
    torch.manual_seed(0)
    w = torch.randn(*shape, dtype=dtype)

    eager_fp8, eager_inv = _to_fp8(w, torch.float8_e4m3fn)
    compiled_fp8, compiled_inv = _to_fp8_compiled(w, torch.float8_e4m3fn)

    # fp8 tensors have no equality op, so compare the raw bytes.
    assert torch.equal(eager_fp8.view(torch.uint8), compiled_fp8.view(torch.uint8))
    assert torch.equal(eager_inv, compiled_inv)


def test_quantization_is_deterministic():
    """The cache is only sound because quantizing the same weight twice agrees."""
    torch.manual_seed(0)
    w = torch.randn(1536, 1536, dtype=torch.bfloat16)

    first_fp8, first_inv = _to_fp8_compiled(w, torch.float8_e4m3fn)
    second_fp8, second_inv = _to_fp8_compiled(w, torch.float8_e4m3fn)

    assert torch.equal(first_fp8.view(torch.uint8), second_fp8.view(torch.uint8))
    assert torch.equal(first_inv, second_inv)
