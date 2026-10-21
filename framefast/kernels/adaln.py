"""Fused LayerNorm + adaptive modulation ("AdaLN-Zero") Triton kernel for CogVideoX.

Every CogVideoX transformer block calls ``CogVideoXLayerNormZero`` twice (before attention and
before the MLP), and each call applies, to both the video and the text tokens::

    y = LayerNorm(x; weight, bias, eps) * (1 + scale)[:, None, :] + shift[:, None, :]

In eager PyTorch that is three kernels over the full ``(B, N, D)`` activation —
``native_layer_norm``, ``mul`` and ``add`` — each reading and writing the whole tensor through
HBM (plus a tiny ``1 + scale``). All of them have an arithmetic intensity of a handful of FLOPs
per byte, far below any GPU's ridge point, so they are purely memory-bound: their run time is
bytes moved / bandwidth. The fused kernel reads ``x`` once and writes ``y`` once (``scale``,
``shift``, ``weight`` and ``bias`` are ``D``-sized and stay in L2/L1), which cuts the HBM traffic
of this op chain from ~6 activation passes to 2. The statistics are accumulated in fp32
regardless of the I/O dtype.

Correctness is tested against :func:`ln_modulate_reference`; on machines without a GPU the tests
run the *same* Triton kernel through the Triton interpreter (``TRITON_INTERPRET=1``).

This is an inference-only kernel (no backward).
"""

from __future__ import annotations

import os
import types
from typing import Any

import torch
import torch.nn.functional as F

try:  # Triton is optional: CPU-only installs fall back to the reference path.
    import triton
    import triton.language as tl

    HAS_TRITON = True
except Exception:  # pragma: no cover - depends on the environment
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    HAS_TRITON = False


def triton_interpret_enabled() -> bool:
    """Whether Triton kernels run in the CPU interpreter (``TRITON_INTERPRET=1``)."""
    return os.environ.get("TRITON_INTERPRET", "0") not in ("", "0")


def ln_modulate_reference(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Exact PyTorch semantics of the CogVideoX op chain (what diffusers computes in eager)."""
    y = F.layer_norm(x, (x.shape[-1],), weight, bias, eps)
    return y * (1 + scale)[:, None, :] + shift[:, None, :]


if HAS_TRITON:

    @triton.jit
    def _ln_modulate_kernel(
        X,
        Y,
        W,
        B,
        SCALE,
        SHIFT,
        stride_xb,
        stride_xn,
        stride_yb,
        stride_yn,
        stride_mb_scale,
        stride_mb_shift,
        D,
        eps,
        HAS_W: tl.constexpr,
        HAS_B: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """One program normalises one token row ``x[b, n, :]`` (grid = (N, B))."""
        n = tl.program_id(0)
        b = tl.program_id(1)
        cols = tl.arange(0, BLOCK_D)
        mask = cols < D

        x = tl.load(X + b * stride_xb + n * stride_xn + cols, mask=mask, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / D
        xc = tl.where(mask, x - mean, 0.0)
        var = tl.sum(xc * xc, axis=0) / D
        rstd = 1.0 / tl.sqrt(var + eps)
        y = xc * rstd
        if HAS_W:
            y = y * tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_B:
            y = y + tl.load(B + cols, mask=mask, other=0.0).to(tl.float32)
        sc = tl.load(SCALE + b * stride_mb_scale + cols, mask=mask, other=0.0).to(tl.float32)
        sh = tl.load(SHIFT + b * stride_mb_shift + cols, mask=mask, other=0.0).to(tl.float32)
        y = y * (1.0 + sc) + sh
        tl.store(Y + b * stride_yb + n * stride_yn + cols, y.to(Y.dtype.element_ty), mask=mask)


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float,
    wrap: Any = None,
) -> torch.Tensor:
    bsz, n_tok, d = x.shape
    if x.stride(-1) != 1:
        x = x.contiguous()
    if scale.stride(-1) != 1:
        scale = scale.contiguous()
    if shift.stride(-1) != 1:
        shift = shift.contiguous()
    y = torch.empty((bsz, n_tok, d), dtype=x.dtype, device=x.device)
    block_d = triton.next_power_of_2(d)
    num_warps = 4 if block_d <= 1024 else (8 if block_d <= 4096 else 16)
    kernel = wrap(_ln_modulate_kernel) if wrap is not None else _ln_modulate_kernel
    # Pointers for absent weight/bias are never dereferenced (HAS_W / HAS_B are constexpr).
    kernel[(n_tok, bsz)](
        x,
        y,
        weight if weight is not None else x,
        bias if bias is not None else x,
        scale,
        shift,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        scale.stride(0),
        shift.stride(0),
        d,
        eps,
        HAS_W=weight is not None,
        HAS_B=bias is not None,
        BLOCK_D=block_d,
        num_warps=num_warps,
    )
    return y


_triton_op: Any = None
if HAS_TRITON and hasattr(torch.library, "triton_op"):
    # Registering as a ``triton_op`` makes the kernel visible to torch.compile/inductor (no graph
    # break), so ``compile+triton_adaln`` stacks cleanly.
    try:
        from torch.library import triton_op, wrap_triton

        @triton_op("framefast::ln_modulate", mutates_args=())
        def _ln_modulate_op(
            x: torch.Tensor,
            weight: torch.Tensor | None,
            bias: torch.Tensor | None,
            scale: torch.Tensor,
            shift: torch.Tensor,
            eps: float,
        ) -> torch.Tensor:
            return _launch(x, weight, bias, scale, shift, eps, wrap=wrap_triton)

        _triton_op = _ln_modulate_op
    except Exception:  # pragma: no cover - older torch
        _triton_op = None


def _validate(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> None:
    if x.dim() != 3:
        raise ValueError(f"x must be (B, N, D), got {tuple(x.shape)}")
    bsz, _, d = x.shape
    for name, t in (("scale", scale), ("shift", shift)):
        if t.shape != (bsz, d):
            raise ValueError(f"{name} must be (B, D)=({bsz}, {d}), got {tuple(t.shape)}")


def ln_modulate(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    scale: torch.Tensor,
    shift: torch.Tensor,
    eps: float = 1e-5,
    backend: str = "auto",
) -> torch.Tensor:
    """Compute ``LayerNorm(x) * (1 + scale)[:, None] + shift[:, None]`` in one pass.

    Args:
        x: ``(B, N, D)`` activations (any float dtype; last dim must be unit-stride or is copied).
        weight, bias: Optional LayerNorm affine parameters of shape ``(D,)``.
        scale, shift: ``(B, D)`` modulation (strided views from ``chunk`` are fine).
        eps: LayerNorm epsilon.
        backend: ``"triton"`` (GPU, or CPU under ``TRITON_INTERPRET=1``), ``"reference"``, or
            ``"auto"`` (Triton on CUDA tensors, reference otherwise).
    """
    _validate(x, scale, shift)
    if backend == "auto":
        backend = "triton" if (HAS_TRITON and x.is_cuda) else "reference"
    if backend == "reference":
        return ln_modulate_reference(x, weight, bias, scale, shift, eps)
    if backend != "triton":
        raise ValueError(f"unknown backend {backend!r}")
    if not HAS_TRITON:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda and not triton_interpret_enabled():
        raise RuntimeError("Triton backend on CPU tensors requires TRITON_INTERPRET=1")
    if x.is_cuda and _triton_op is not None:
        return _triton_op(x, weight, bias, scale, shift, eps)
    return _launch(x, weight, bias, scale, shift, eps)


def _fused_layernorm_zero_forward(
    self: Any, hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor, temb: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Drop-in replacement for ``CogVideoXLayerNormZero.forward`` using :func:`ln_modulate`."""
    shift, scale, gate, enc_shift, enc_scale, enc_gate = self.linear(self.silu(temb)).chunk(6, dim=1)
    norm = self.norm
    backend = self._framefast_backend
    hs = ln_modulate(hidden_states, norm.weight, norm.bias, scale, shift, norm.eps, backend)
    ehs = ln_modulate(encoder_hidden_states, norm.weight, norm.bias, enc_scale, enc_shift, norm.eps, backend)
    return hs, ehs, gate[:, None, :], enc_gate[:, None, :]


def patch_cogvideox_adaln(model: torch.nn.Module, backend: str = "auto") -> int:
    """Swap the forward of every ``CogVideoXLayerNormZero`` in ``model`` for the fused kernel.

    Returns:
        Number of modules patched (2 per transformer block). Weights are untouched, so the
        patch is reversible with :func:`unpatch_cogvideox_adaln`.
    """
    from diffusers.models.normalization import CogVideoXLayerNormZero

    count = 0
    for mod in model.modules():
        if isinstance(mod, CogVideoXLayerNormZero):
            mod._framefast_backend = backend
            mod.forward = types.MethodType(_fused_layernorm_zero_forward, mod)
            count += 1
    return count


def unpatch_cogvideox_adaln(model: torch.nn.Module) -> int:
    """Undo :func:`patch_cogvideox_adaln`; returns the number of modules restored."""
    count = 0
    for mod in model.modules():
        if "_framefast_backend" in vars(mod):
            del mod.forward
            del mod._framefast_backend
            count += 1
    return count
