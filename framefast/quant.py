"""Weight quantization of the denoising transformer.

Two implementations are provided:

* **torchao** (preferred on GPU): ``Int8WeightOnlyConfig``, ``Int8DynamicActivationInt8WeightConfig``,
  ``Float8WeightOnlyConfig`` and ``Float8DynamicActivationFloat8WeightConfig`` applied with
  ``torchao.quantization.quantize_``. These produce tensor subclasses whose matmuls lower to
  fused int8/fp8 kernels, especially under ``torch.compile``.
* **Reference INT8 weight-only** (:class:`Int8WeightOnlyLinear`, pure PyTorch): symmetric
  per-output-channel int8 weights with an fp scale, dequantized in ``forward``. It runs anywhere
  (including the CPU tests) and is mathematically the same scheme torchao's int8 weight-only uses,
  but in eager mode it materializes the dequantized weight on every call, so it saves weight
  *memory* but is not expected to be faster. It exists so the quantization error can be
  unit-tested without a GPU.

Why weight-only helps: at the batch sizes of video diffusion the transformer's linear layers
are large GEMMs ``(B·N, D) x (D, 4D)`` with ``B·N`` in the tens of thousands, i.e. mostly
compute-bound, so halving weight bytes (fp16 -> int8) mainly cuts memory footprint; the latency
benefit of weight-only is largest for small-M (memory-bound) matmuls such as the timestep/text
projections. Dynamic-activation int8/fp8 quantizes *both* operands and uses int8/fp8 tensor
cores, which raises peak compute throughput — the lever that matters for compute-bound GEMMs —
at the cost of activation-quantization error. FP8 tensor-core GEMMs need compute capability
>= 8.9 (Ada / Hopper).
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

TORCHAO_SCHEMES = ("int8_wo", "int8_dyn", "fp8_wo", "fp8_dyn")


def quantize_per_channel_int8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-output-channel int8 quantization.

    Args:
        weight: ``(out_features, in_features)`` float weight.

    Returns:
        ``(q, scale)`` with ``q`` int8 in ``[-127, 127]`` and ``scale`` of shape ``(out, 1)``
        (fp32) such that ``q * scale ≈ weight``.
    """
    w = weight.detach().float()
    amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = amax / 127.0
    q = torch.clamp(torch.round(w / scale), -127, 127).to(torch.int8)
    return q, scale


class Int8WeightOnlyLinear(nn.Module):
    """Drop-in ``nn.Linear`` replacement storing int8 weights + per-channel scales."""

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__()
        q, scale = quantize_per_channel_int8(linear.weight)
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer("qweight", q)
        self.register_buffer("scale", scale.to(linear.weight.dtype))
        self.bias = None if linear.bias is None else nn.Parameter(linear.bias.detach().clone(), False)

    @property
    def weight(self) -> torch.Tensor:
        """Dequantized weight (computed on access; used by code that inspects ``.weight``)."""
        return self.qweight.to(self.scale.dtype) * self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x @ (q * s).T + b`` — the dequantize is fusable into the GEMM by inductor."""
        w = self.qweight.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, int8 per-channel"


LinearFilter = Callable[[nn.Module, str], bool]


def default_filter(min_features: int = 16) -> LinearFilter:
    """Quantize every ``nn.Linear`` whose both dims are at least ``min_features``.

    Very narrow projections gain nothing from int8 storage and lose relatively more accuracy.
    (All linears of CogVideoX-2B are >= 512 wide, so the default quantizes all of them.)
    """

    def _f(mod: nn.Module, fqn: str) -> bool:
        # torchao calls the filter on *every* module, so check the type here.
        return (
            isinstance(mod, nn.Linear)
            and mod.in_features >= min_features
            and mod.out_features >= min_features
        )

    return _f


def apply_reference_int8(model: nn.Module, filter_fn: LinearFilter | None = None) -> int:
    """Replace eligible ``nn.Linear`` modules in-place with :class:`Int8WeightOnlyLinear`.

    Returns:
        Number of layers replaced.
    """
    filter_fn = filter_fn or default_filter()
    targets: list[tuple[nn.Module, str, nn.Linear]] = []
    for parent_name, parent in model.named_modules():
        for child_name, child in parent.named_children():
            fqn = f"{parent_name}.{child_name}" if parent_name else child_name
            if type(child) is nn.Linear and filter_fn(child, fqn):
                targets.append((parent, child_name, child))
    for parent, name, child in targets:
        setattr(parent, name, Int8WeightOnlyLinear(child))
    return len(targets)


def fp8_supported() -> tuple[bool, str]:
    """Whether FP8 tensor-core GEMMs are available on the current device."""
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < (8, 9):
        return False, f"FP8 needs compute capability >= 8.9, device is {major}.{minor}"
    return True, ""


def apply_torchao(model: nn.Module, scheme: str, filter_fn: LinearFilter | None = None) -> None:
    """Quantize ``model`` in place with torchao.

    Args:
        model: Usually ``pipe.transformer``.
        scheme: One of :data:`TORCHAO_SCHEMES`.
        filter_fn: Which linears to quantize (defaults to :func:`default_filter`).
    """
    from torchao.quantization import quantize_

    filter_fn = filter_fn or default_filter()
    try:  # torchao >= 0.9 config-object API
        from torchao.quantization import (
            Float8DynamicActivationFloat8WeightConfig,
            Float8WeightOnlyConfig,
            Int8DynamicActivationInt8WeightConfig,
            Int8WeightOnlyConfig,
        )

        configs = {
            "int8_wo": Int8WeightOnlyConfig,
            "int8_dyn": Int8DynamicActivationInt8WeightConfig,
            "fp8_wo": Float8WeightOnlyConfig,
            "fp8_dyn": Float8DynamicActivationFloat8WeightConfig,
        }
        if scheme not in configs:
            raise ValueError(f"unknown scheme {scheme!r}")
        cfg = configs[scheme]()
        if scheme == "fp8_dyn":
            try:
                from torchao.quantization import PerRow

                cfg = Float8DynamicActivationFloat8WeightConfig(granularity=PerRow())
            except Exception:
                pass
    except ImportError:  # older torchao function API
        from torchao.quantization import (  # type: ignore[attr-defined,no-redef]
            float8_dynamic_activation_float8_weight,
            float8_weight_only,
            int8_dynamic_activation_int8_weight,
            int8_weight_only,
        )

        fns = {
            "int8_wo": int8_weight_only,
            "int8_dyn": int8_dynamic_activation_int8_weight,
            "fp8_wo": float8_weight_only,
            "fp8_dyn": float8_dynamic_activation_float8_weight,
        }
        if scheme not in fns:
            raise ValueError(f"unknown scheme {scheme!r}") from None
        cfg = fns[scheme]()
    quantize_(model, cfg, filter_fn=filter_fn)


def module_bytes(model: nn.Module) -> int:
    """Bytes held by parameters + buffers of the model (quantized storage included)."""
    total = 0
    seen: set[int] = set()
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total
