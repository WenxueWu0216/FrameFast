"""Numerical correctness of the fused LayerNorm+modulation kernel against the PyTorch reference.

On CPU-only machines the Triton kernel itself is executed by the Triton interpreter
(TRITON_INTERPRET=1 is set in conftest.py); the CUDA tests run the compiled kernel.
"""

import pytest
import torch

from framefast.kernels import adaln
from framefast.kernels.adaln import (
    HAS_TRITON,
    ln_modulate,
    ln_modulate_reference,
    patch_cogvideox_adaln,
    unpatch_cogvideox_adaln,
)
from tests.conftest import requires_cuda

needs_triton = pytest.mark.skipif(not HAS_TRITON, reason="triton not installed")


def _inputs(b, n, d, dtype, device="cpu", affine=True, strided_mod=True, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(b, n, d, generator=g) * 3 + 1).to(device, dtype)
    w = (torch.randn(d, generator=g) * 0.5 + 1).to(device, dtype) if affine else None
    bb = (torch.randn(d, generator=g) * 0.1).to(device, dtype) if affine else None
    if strided_mod:  # exactly how CogVideoX produces them: chunk() views of a (B, 6D) tensor
        emb = torch.randn(b, 6 * d, generator=g).to(device, dtype)
        shift, scale = emb.chunk(6, dim=1)[:2]
    else:
        shift = torch.randn(b, d, generator=g).to(device, dtype)
        scale = torch.randn(b, d, generator=g).to(device, dtype)
    return x, w, bb, scale, shift


def _ref_fp32(x, w, b, scale, shift, eps=1e-5):
    f = lambda t: None if t is None else t.float()  # noqa: E731
    return ln_modulate_reference(x.float(), f(w), f(b), scale.float(), shift.float(), eps)


@needs_triton
@pytest.mark.parametrize(
    "shape", [(1, 5, 32), (2, 7, 48), (2, 3, 130), (1, 1, 1920)], ids=lambda s: "x".join(map(str, s))
)
@pytest.mark.parametrize("affine", [True, False])
@pytest.mark.parametrize("strided_mod", [True, False])
def test_triton_interpreter_matches_reference_fp32(shape, affine, strided_mod):
    if not adaln.triton_interpret_enabled() and not torch.cuda.is_available():
        pytest.skip("needs TRITON_INTERPRET=1 or CUDA")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x, w, b, scale, shift = _inputs(*shape, torch.float32, device, affine, strided_mod)
    out = ln_modulate(x, w, b, scale, shift, 1e-5, backend="triton")
    ref = ln_modulate_reference(x, w, b, scale, shift, 1e-5)
    assert out.dtype == x.dtype and out.shape == x.shape
    torch.testing.assert_close(out, ref, atol=2e-5, rtol=2e-5)


@needs_triton
def test_triton_handles_non_contiguous_token_slice():
    """CogVideoX passes ``hidden_states[:, text_len:]`` — a strided view of the token axis."""
    if not adaln.triton_interpret_enabled() and not torch.cuda.is_available():
        pytest.skip("needs TRITON_INTERPRET=1 or CUDA")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, w, b, scale, shift = _inputs(2, 11, 64, torch.float32, device)
    x = full[:, 3:]
    assert not x.is_contiguous()
    torch.testing.assert_close(
        ln_modulate(x, w, b, scale, shift, backend="triton"),
        ln_modulate_reference(x, w, b, scale, shift, 1e-5),
        atol=2e-5,
        rtol=2e-5,
    )


def test_reference_backend_is_exact_eager_semantics():
    x, w, b, scale, shift = _inputs(2, 4, 16, torch.float32)
    expected = torch.nn.functional.layer_norm(x, (16,), w, b, 1e-5) * (1 + scale[:, None]) + shift[:, None]
    torch.testing.assert_close(ln_modulate(x, w, b, scale, shift, backend="reference"), expected)


def test_input_validation():
    x, w, b, scale, shift = _inputs(2, 4, 16, torch.float32)
    with pytest.raises(ValueError):
        ln_modulate(x[0], w, b, scale, shift)
    with pytest.raises(ValueError):
        ln_modulate(x, w, b, scale[:1], shift)
    with pytest.raises(ValueError):
        ln_modulate(x, w, b, scale, shift, backend="nope")


def test_patched_tiny_transformer_matches_unpatched(tiny_pipe):
    tr = tiny_pipe.transformer.eval()
    g = torch.Generator().manual_seed(0)
    kw = dict(
        hidden_states=torch.randn(2, 3, 4, 4, 4, generator=g),
        encoder_hidden_states=torch.randn(2, 16, 32, generator=g),
        timestep=torch.tensor([500, 500]),
        return_dict=False,
    )
    with torch.no_grad():
        ref = tr(**kw)[0]
        backend = "triton" if HAS_TRITON and adaln.triton_interpret_enabled() else "reference"
        assert patch_cogvideox_adaln(tr, backend=backend) == 2 * len(tr.transformer_blocks)
        out = tr(**kw)[0]
        assert unpatch_cogvideox_adaln(tr) == 2 * len(tr.transformer_blocks)
        again = tr(**kw)[0]
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(again, ref, atol=0, rtol=0)


# --------------------------------------------------------------------------- GPU-only tests
@requires_cuda
@needs_triton
@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(2, 17550, 1920), (2, 226, 1920), (1, 333, 3072)])
def test_cuda_half_precision_matches_fp32_reference(dtype, shape):
    """The fused kernel accumulates in fp32, so it should be at least as close to an fp32
    reference as the eager fp16/bf16 op chain is."""
    x, w, b, scale, shift = _inputs(*shape, dtype, "cuda")
    ref32 = _ref_fp32(x, w, b, scale, shift)
    fused = ln_modulate(x, w, b, scale, shift, backend="triton")
    eager = ln_modulate_reference(x, w, b, scale, shift, 1e-5)
    err_fused = (fused.float() - ref32).abs().max().item()
    err_eager = (eager.float() - ref32).abs().max().item()
    tol = 2e-2 if dtype == torch.float16 else 1.5e-1
    assert err_fused <= max(err_eager * 1.01, tol / 10) and err_fused < tol


@requires_cuda
@needs_triton
@pytest.mark.gpu
def test_cuda_triton_op_under_torch_compile():
    x, w, b, scale, shift = _inputs(2, 64, 256, torch.float16, "cuda")
    fn = torch.compile(lambda *a: ln_modulate(*a, backend="triton"), fullgraph=True)
    torch.testing.assert_close(
        fn(x, w, b, scale, shift, 1e-5), ln_modulate(x, w, b, scale, shift), atol=0, rtol=0
    )
