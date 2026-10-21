"""The optimization ladder: a registry of independently-measurable pipeline variants.

A variant spec is a ``+``-joined list of registered names, e.g. ``"compile+int8_wo"``; the parts
are applied left to right to a freshly loaded pipeline. Every variant is compared against
``baseline`` (fp16 eager, default SDPA dispatch) from the same run.

Each :class:`Variant` declares its requirements (CUDA, Python packages, device capability) so
the runner can mark it *skipped* with a reason instead of failing — e.g. ``fp8_dyn`` on an A100.
"""

from __future__ import annotations

import contextlib
import importlib.util
import re
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any

import torch

RUNGS: dict[str, str] = {
    "0": "baseline",
    "1": "torch.compile",
    "2": "attention",
    "3": "quantization",
    "4": "custom Triton kernel",
    "5": "TensorRT",
    "6": "memory / batching",
}


@dataclass
class RunContext:
    """Information a variant may need while being applied."""

    device: torch.device
    dtype: torch.dtype
    call_kwargs: dict[str, Any]
    is_tiny: bool = False


ApplyFn = Callable[[Any, RunContext], list[str]]


@dataclass
class Variant:
    """One rung (or sub-rung) of the ladder.

    Attributes:
        name: Registry key.
        rung: Key into :data:`RUNGS`.
        description: One line shown in the report.
        apply: Mutates the pipeline in place; returns notes for the report.
        requires: Importable modules that must be present.
        requires_cuda: Skip when no CUDA device is present.
        check: Extra availability check returning a skip reason (or ``None``).
        placement: ``"device"``, ``"model_offload"`` or ``"sequential_offload"``.
        call_context: Context manager factory wrapped around every pipeline call.
        batch_size: Number of videos per call (prompt list); ``None`` = 1.
        max_runs: Cap on timed runs (for very slow variants such as sequential offload).
        apply_last: Apply after all other parts of a stack regardless of spec order. Set for
            ``torch.compile`` so module rewrites (quantization, kernel patches) are traced.
    """

    name: str
    rung: str
    description: str
    apply: ApplyFn | None = None
    requires: tuple[str, ...] = ()
    requires_cuda: bool = False
    check: Callable[[], str | None] | None = None
    placement: str = "device"
    call_context: Callable[[], AbstractContextManager[Any]] | None = None
    batch_size: int | None = None
    max_runs: int | None = None
    apply_last: bool = False


@dataclass
class ResolvedVariant:
    """A (possibly stacked) variant ready to run."""

    spec: str
    parts: list[Variant] = field(default_factory=list)

    @property
    def rung(self) -> str:
        return max(p.rung for p in self.parts)

    @property
    def description(self) -> str:
        return " + ".join(p.description for p in self.parts)

    @property
    def placement(self) -> str:
        places = {p.placement for p in self.parts} - {"device"}
        if len(places) > 1:
            raise ValueError(f"{self.spec}: conflicting placements {places}")
        return places.pop() if places else "device"

    @property
    def batch_size(self) -> int:
        sizes = [p.batch_size for p in self.parts if p.batch_size is not None]
        if len(sizes) > 1:
            raise ValueError(f"{self.spec}: more than one batch size")
        return sizes[0] if sizes else 1

    @property
    def max_runs(self) -> int | None:
        caps = [p.max_runs for p in self.parts if p.max_runs is not None]
        return min(caps) if caps else None

    def skip_reason(self) -> str | None:
        """Why this variant cannot run here, or ``None`` if it can."""
        for p in self.parts:
            if p.requires_cuda and not torch.cuda.is_available():
                return f"{p.name}: requires a CUDA GPU"
            for mod in p.requires:
                if importlib.util.find_spec(mod) is None:
                    return f"{p.name}: Python package '{mod}' is not installed"
            if p.check is not None and (reason := p.check()):
                return f"{p.name}: {reason}"
        return None

    @contextlib.contextmanager
    def call_context(self) -> Iterator[None]:
        """Nest the call contexts of all parts."""
        with contextlib.ExitStack() as stack:
            for p in self.parts:
                if p.call_context is not None:
                    stack.enter_context(p.call_context())
            yield

    def apply(self, pipe: Any, ctx: RunContext) -> list[str]:
        """Apply every part in order and collect their notes."""
        notes: list[str] = []
        for p in sorted(self.parts, key=lambda v: v.apply_last):  # stable: keeps spec order otherwise
            if p.apply is not None:
                notes.extend(p.apply(pipe, ctx))
        return notes


# ----------------------------------------------------------------------------- implementations
def _pin_pos_embed(pipe: Any, ctx: RunContext) -> list[str]:
    """Avoid diffusers' per-call NumPy positional-embedding rebuild at non-native resolutions."""
    from framefast.trt import capture_transformer_inputs, pin_positional_embeddings

    cap = capture_transformer_inputs(pipe, dict(ctx.call_kwargs))
    if pin_positional_embeddings(pipe.transformer, tuple(cap["hidden_states"].shape)):
        return ["pinned sin-cos positional embedding (non-native resolution)"]
    return []


def _compile(mode: str) -> ApplyFn:
    def apply(pipe: Any, ctx: RunContext) -> list[str]:
        notes = _pin_pos_embed(pipe, ctx)
        if ctx.device.type == "cuda":
            import torch._inductor.config as ic

            # Settings recommended by the PyTorch team for diffusion transformers.
            ic.conv_1x1_as_mm = True
            ic.coordinate_descent_tuning = True
            ic.epilogue_fusion = False
            ic.coordinate_descent_check_all_directions = True
        pipe.transformer = torch.compile(pipe.transformer, mode=mode, fullgraph=False)
        notes.append(f"torch.compile(transformer, mode={mode!r})")
        return notes

    return apply


def _sdpa(backend_name: str) -> Callable[[], AbstractContextManager[Any]]:
    def ctx() -> AbstractContextManager[Any]:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        return sdpa_kernel([getattr(SDPBackend, backend_name)])

    return ctx


def _sdpa_check(backend_name: str) -> Callable[[], str | None]:
    def check() -> str | None:
        try:
            from torch.nn.attention import SDPBackend
        except ImportError:
            return "torch.nn.attention.SDPBackend unavailable (torch too old)"
        if not hasattr(SDPBackend, backend_name):
            return f"SDPBackend.{backend_name} not available in this torch build"
        return None

    return check


def _fuse_qkv(pipe: Any, ctx: RunContext) -> list[str]:
    pipe.fuse_qkv_projections()
    return ["fused Q/K/V projections into one GEMM per attention layer"]


def _torchao(scheme: str) -> ApplyFn:
    def apply(pipe: Any, ctx: RunContext) -> list[str]:
        from framefast.quant import apply_torchao, module_bytes

        before = module_bytes(pipe.transformer)
        apply_torchao(pipe.transformer, scheme)
        return [f"torchao {scheme} on transformer linears (fp weights were {before / 1e6:.1f} MB)"]

    return apply


def _int8_ref(pipe: Any, ctx: RunContext) -> list[str]:
    from framefast.quant import apply_reference_int8, module_bytes

    before = module_bytes(pipe.transformer)
    n = apply_reference_int8(pipe.transformer)
    after = module_bytes(pipe.transformer)
    return [
        f"reference int8 weight-only: {n} linears, transformer {before / 1e6:.2f} MB -> {after / 1e6:.2f} MB"
    ]


def _fp8_check() -> str | None:
    from framefast.quant import fp8_supported

    ok, why = fp8_supported()
    return None if ok else why


def _triton_check() -> str | None:
    from framefast.kernels.adaln import HAS_TRITON, triton_interpret_enabled

    if not HAS_TRITON:
        return "triton is not installed"
    if not torch.cuda.is_available() and not triton_interpret_enabled():
        return "no CUDA device (set TRITON_INTERPRET=1 to run the kernel in the CPU interpreter)"
    return None


def _triton_adaln(pipe: Any, ctx: RunContext) -> list[str]:
    from framefast.kernels.adaln import patch_cogvideox_adaln

    n = patch_cogvideox_adaln(pipe.transformer, backend="triton")
    if n == 0:
        raise RuntimeError("no CogVideoXLayerNormZero modules found (not a CogVideoX transformer?)")
    note = f"fused LayerNorm+modulation Triton kernel in {n} CogVideoXLayerNormZero modules"
    if ctx.device.type == "cpu":
        note += " (Triton interpreter on CPU — correctness only, not a performance number)"
    return [note]


def _trt_transformer(pipe: Any, ctx: RunContext) -> list[str]:
    from framefast.trt import compile_transformer_trt

    return compile_transformer_trt(pipe, dict(ctx.call_kwargs), ctx.dtype)


def _trt_vae(pipe: Any, ctx: RunContext) -> list[str]:
    """Export ``vae.decode`` for the run's latent shape and compile it with Torch-TensorRT."""
    import torch_tensorrt  # type: ignore[import-not-found]
    from diffusers.models.autoencoders.vae import DecoderOutput

    from framefast.trt import VAEDecodeWrapper, capture_transformer_inputs, probe_export

    cap = capture_transformer_inputs(pipe, dict(ctx.call_kwargs))
    b2, f, c, h, w = cap["hidden_states"].shape
    batch = b2 // 2 if ctx.call_kwargs.get("guidance_scale", 1.0) > 1.0 else b2
    z = torch.randn(batch, c, f, h, w, device=ctx.device, dtype=pipe.vae.dtype)
    probe, ep = probe_export("vae.decode", VAEDecodeWrapper(pipe.vae).eval(), (z,))
    notes = [probe.summary()]
    if ep is None:
        raise RuntimeError(probe.summary())
    compiled = torch_tensorrt.dynamo.compile(
        ep, arg_inputs=[z], enabled_precisions={ctx.dtype}, truncate_double=True, min_block_size=1
    )
    expected = tuple(z.shape)

    def decode(latents: torch.Tensor, return_dict: bool = True) -> Any:
        if tuple(latents.shape) != expected:
            raise RuntimeError(f"TRT VAE built for {expected}, got {tuple(latents.shape)}")
        out = compiled(latents)
        return DecoderOutput(sample=out) if return_dict else (out,)

    pipe.vae.decode = decode
    notes.append("vae.decode replaced by a static-shape Torch-TensorRT module")
    return notes


def _vae_tiling(pipe: Any, ctx: RunContext) -> list[str]:
    pipe.vae.enable_tiling()
    return ["VAE decodes in overlapping spatial tiles (lower peak memory)"]


def _vae_slicing(pipe: Any, ctx: RunContext) -> list[str]:
    pipe.vae.enable_slicing()
    return ["VAE decodes one video of the batch at a time"]


def _noop(note: str) -> ApplyFn:
    return lambda pipe, ctx: [note]


_REGISTRY: dict[str, Variant] = {}


def register(v: Variant) -> Variant:
    """Add a variant to the registry (name must be unique)."""
    if v.name in _REGISTRY:
        raise ValueError(f"duplicate variant {v.name}")
    _REGISTRY[v.name] = v
    return v


for _v in (
    Variant("baseline", "0", "fp16 eager PyTorch, default SDPA dispatch"),
    Variant(
        "compile",
        "1",
        "torch.compile transformer (inductor, max-autotune, no CUDA graphs)",
        apply=_compile("max-autotune-no-cudagraphs"),
        apply_last=True,
    ),
    Variant(
        "compile_cudagraphs",
        "1",
        "torch.compile transformer (max-autotune incl. CUDA graphs)",
        apply=_compile("max-autotune"),
        requires_cuda=True,
        apply_last=True,
    ),
    Variant(
        "attn_math",
        "2",
        "SDPA forced to MATH backend (unfused reference: materializes N×N scores)",
        call_context=_sdpa("MATH"),
        check=_sdpa_check("MATH"),
    ),
    Variant(
        "attn_flash",
        "2",
        "SDPA forced to FlashAttention-2 backend",
        call_context=_sdpa("FLASH_ATTENTION"),
        check=_sdpa_check("FLASH_ATTENTION"),
        requires_cuda=True,
    ),
    Variant(
        "attn_efficient",
        "2",
        "SDPA forced to memory-efficient (xFormers-style) backend",
        call_context=_sdpa("EFFICIENT_ATTENTION"),
        check=_sdpa_check("EFFICIENT_ATTENTION"),
        requires_cuda=True,
    ),
    Variant(
        "attn_cudnn",
        "2",
        "SDPA forced to cuDNN attention backend",
        call_context=_sdpa("CUDNN_ATTENTION"),
        check=_sdpa_check("CUDNN_ATTENTION"),
        requires_cuda=True,
    ),
    Variant("fuse_qkv", "2", "fused QKV projection", apply=_fuse_qkv),
    Variant(
        "int8_wo",
        "3",
        "torchao INT8 weight-only",
        apply=_torchao("int8_wo"),
        requires=("torchao",),
    ),
    Variant(
        "int8_dyn",
        "3",
        "torchao INT8 dynamic-activation + INT8 weight",
        apply=_torchao("int8_dyn"),
        requires=("torchao",),
    ),
    Variant(
        "fp8_wo",
        "3",
        "torchao FP8 (e4m3) weight-only",
        apply=_torchao("fp8_wo"),
        requires=("torchao",),
        requires_cuda=True,
        check=_fp8_check,
    ),
    Variant(
        "fp8_dyn",
        "3",
        "torchao FP8 dynamic-activation + FP8 weight (per-row)",
        apply=_torchao("fp8_dyn"),
        requires=("torchao",),
        requires_cuda=True,
        check=_fp8_check,
    ),
    Variant("int8_wo_ref", "3", "reference INT8 weight-only (pure PyTorch dequant)", apply=_int8_ref),
    Variant(
        "triton_adaln",
        "4",
        "custom Triton fused LayerNorm+AdaLN modulation",
        apply=_triton_adaln,
        check=_triton_check,
    ),
    Variant(
        "trt_transformer",
        "5",
        "Torch-TensorRT engine for the transformer (static shapes)",
        apply=_trt_transformer,
        requires=("torch_tensorrt",),
        requires_cuda=True,
    ),
    Variant(
        "trt_vae",
        "5",
        "Torch-TensorRT engine for vae.decode (static shapes)",
        apply=_trt_vae,
        requires=("torch_tensorrt",),
        requires_cuda=True,
    ),
    Variant("vae_tiling", "6", "VAE spatial tiling", apply=_vae_tiling),
    Variant("vae_slicing", "6", "VAE batch slicing", apply=_vae_slicing),
    Variant(
        "cpu_offload_model",
        "6",
        "model-level CPU offload (accelerate hooks)",
        apply=_noop("enable_model_cpu_offload: whole sub-models move to GPU on use"),
        placement="model_offload",
        requires_cuda=True,
        requires=("accelerate",),
    ),
    Variant(
        "cpu_offload_seq",
        "6",
        "sequential (per-layer) CPU offload",
        apply=_noop("enable_sequential_cpu_offload: weights stream per layer"),
        placement="sequential_offload",
        requires_cuda=True,
        requires=("accelerate",),
        max_runs=2,
    ),
):
    register(_v)

_BATCH_RE = re.compile(r"^bs(\d+)$")


def get_variant(name: str) -> Variant:
    """Look up a registered variant; ``bs<N>`` is synthesised on demand (batch of N videos)."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    m = _BATCH_RE.match(name)
    if m:
        n = int(m.group(1))
        if n < 1:
            raise ValueError("batch size must be >= 1")
        return Variant(name, "6", f"batch of {n} videos per call", batch_size=n)
    raise KeyError(f"unknown variant {name!r}; known: {sorted(_REGISTRY)} or bs<N>")


def resolve(spec: str) -> ResolvedVariant:
    """Parse a ``+``-stacked spec into a :class:`ResolvedVariant`."""
    names = [s.strip() for s in spec.split("+") if s.strip()]
    if not names:
        raise ValueError("empty variant spec")
    rv = ResolvedVariant(spec=spec, parts=[get_variant(n) for n in names])
    _ = rv.placement, rv.batch_size  # validate stacking early
    return rv


def registered_names() -> list[str]:
    """All registered (non-parametric) variant names."""
    return list(_REGISTRY)
