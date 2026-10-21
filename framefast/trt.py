"""TensorRT export (via Torch-TensorRT) of the CogVideoX transformer, plus export probes.

Approach:

1. Run the pipeline until the first transformer call and capture its exact inputs
   (:func:`capture_transformer_inputs`). TensorRT engines here are built for *static* shapes —
   the denoising loop calls the transformer with identical shapes at every step, so one engine
   serves the whole loop.
2. Pin the sin-cos positional embedding for that resolution (:func:`pin_positional_embeddings`):
   diffusers recomputes it with NumPy on every call when the resolution differs from the
   checkpoint's native one, which is untraceable by ``torch.export`` (and a host round-trip).
3. Wrap the transformer in a positional-argument module (:class:`TransformerExportWrapper`),
   ``torch.export`` it, and compile with ``torch_tensorrt.dynamo.compile``. Unsupported ops fall
   back to PyTorch subgraphs; the number of TRT engines vs. fallback subgraphs is reported.
4. Swap ``pipe.transformer`` for :class:`TRTTransformerShim`, which exposes the attributes the
   pipeline reads (``config``, ``dtype``, ``cache_context``) and calls the engine.

The VAE decoder is only *probed* (:func:`probe_export`): see the README for why it is not run
through TensorRT by default.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn


class _Captured(Exception):
    """Raised from a forward pre-hook to stop the pipeline after input capture."""


def capture_transformer_inputs(pipe: Any, call_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return the keyword inputs of the first ``pipe.transformer`` call for ``call_kwargs``."""
    captured: dict[str, Any] = {}

    def hook(module: nn.Module, args: tuple, kwargs: dict[str, Any]) -> None:
        if args:
            raise RuntimeError("expected the pipeline to call the transformer with keyword arguments")
        captured.update(kwargs)
        raise _Captured

    handle = pipe.transformer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        pipe(**call_kwargs)
    except _Captured:
        pass
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError("transformer was never called")
    return captured


def pin_positional_embeddings(transformer: nn.Module, hidden_states_shape: tuple[int, ...]) -> bool:
    """Precompute CogVideoX's sin-cos positional embedding for a latent shape ``(B, F, C, H, W)``.

    Returns:
        ``True`` if the embedding had to be (re)computed, ``False`` if it already matched.
    """
    pe = getattr(transformer, "patch_embed", None)
    if pe is None or not getattr(pe, "use_positional_embeddings", False):
        return False
    if getattr(pe, "use_learned_positional_embeddings", False):
        return False
    _, frames, _, height, width = hidden_states_shape
    pre_frames = (frames - 1) * pe.temporal_compression_ratio + 1
    if (pe.sample_height, pe.sample_width, pe.sample_frames) == (height, width, pre_frames):
        return False
    device = pe.pos_embedding.device
    pe.pos_embedding = pe._get_positional_embeddings(height, width, pre_frames, device=device)
    pe.sample_height, pe.sample_width, pe.sample_frames = height, width, pre_frames
    return True


class TransformerExportWrapper(nn.Module):
    """Positional-argument, tensor-only view of ``CogVideoXTransformer3DModel.forward``."""

    def __init__(self, transformer: nn.Module, has_rotary: bool) -> None:
        super().__init__()
        self.transformer = transformer
        self.has_rotary = has_rotary

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        rope_cos: torch.Tensor | None = None,
        rope_sin: torch.Tensor | None = None,
    ) -> torch.Tensor:
        rotary = (rope_cos, rope_sin) if self.has_rotary else None
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            image_rotary_emb=rotary,
            return_dict=False,
        )[0]


def export_inputs(captured: dict[str, Any]) -> tuple[tuple[torch.Tensor, ...], bool]:
    """Turn captured pipeline kwargs into the positional tuple for :class:`TransformerExportWrapper`."""
    unsupported = [k for k in ("timestep_cond", "ofs", "attention_kwargs") if captured.get(k) is not None]
    if unsupported:
        raise NotImplementedError(f"export of transformer inputs {unsupported} is not implemented")
    t = captured["timestep"]
    if not torch.is_tensor(t):
        t = torch.tensor([t])
    args: list[torch.Tensor] = [captured["hidden_states"], captured["encoder_hidden_states"], t]
    rot = captured.get("image_rotary_emb")
    if rot is not None:
        args.extend(rot)
    return tuple(args), rot is not None


@dataclass
class ExportProbe:
    """Outcome of an attempted ``torch.export`` of a module."""

    target: str
    ok: bool
    seconds: float
    graph_nodes: int | None = None
    error: str | None = None

    def summary(self) -> str:
        """One-line description for the report."""
        if self.ok:
            return f"{self.target}: torch.export OK ({self.graph_nodes} nodes, {self.seconds:.1f}s)"
        return f"{self.target}: torch.export FAILED — {self.error}"


def probe_export(target: str, module: nn.Module, args: tuple[Any, ...]) -> tuple[ExportProbe, Any]:
    """Try ``torch.export.export(module, args, strict=False)``; never raises.

    Returns:
        ``(probe, exported_program_or_None)``.
    """
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            ep = torch.export.export(module, args, strict=False)
        return ExportProbe(target, True, time.perf_counter() - t0, len(ep.graph.nodes)), ep
    except Exception as exc:
        msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:300]}"
        return ExportProbe(target, False, time.perf_counter() - t0, error=msg), None


class VAEDecodeWrapper(nn.Module):
    """``latents -> frames`` view of ``AutoencoderKLCogVideoX.decode`` for export probing."""

    def __init__(self, vae: nn.Module) -> None:
        super().__init__()
        self.vae = vae

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z).sample


class TRTTransformerShim(nn.Module):
    """Stands in for ``pipe.transformer`` and runs a compiled (TRT) static-shape module.

    Only the attributes the CogVideoX pipeline touches are provided.
    """

    def __init__(
        self,
        compiled: Any,
        config: Any,
        dtype: torch.dtype,
        device: torch.device,
        expected_shapes: tuple[torch.Size, ...],
        has_rotary: bool,
    ) -> None:
        super().__init__()
        self.compiled = compiled
        self.config = config
        self._dtype = dtype
        self._device = device
        self.expected_shapes = expected_shapes
        self.has_rotary = has_rotary

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @contextlib.contextmanager
    def cache_context(self, name: str) -> Iterator[None]:
        """No-op (the pipeline's cache-context API; TRT has no diffusers cache hooks)."""
        yield

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        return_dict: bool = False,
        **unused: Any,
    ) -> tuple[torch.Tensor]:
        args: list[torch.Tensor] = [hidden_states, encoder_hidden_states, timestep]
        if self.has_rotary:
            assert image_rotary_emb is not None
            args.extend(image_rotary_emb)
        shapes = tuple(a.shape for a in args)
        if shapes != self.expected_shapes:
            raise RuntimeError(
                f"TRT engine was built for static shapes {self.expected_shapes}, got {shapes}; "
                "rebuild for this resolution/batch"
            )
        return (self.compiled(*args),)


def count_trt_partitions(compiled: Any) -> tuple[int, int]:
    """Count (TensorRT engine submodules, PyTorch fallback submodules) in a torch-trt module."""
    trt, torch_fallback = 0, 0
    for name, _ in getattr(compiled, "named_children", list)():
        if name.startswith("_run_on_acc"):
            trt += 1
        elif name.startswith("_run_on_gpu"):
            torch_fallback += 1
    return trt, torch_fallback


def compile_transformer_trt(pipe: Any, call_kwargs: dict[str, Any], dtype: torch.dtype) -> list[str]:
    """Replace ``pipe.transformer`` with a Torch-TensorRT engine for the shapes of ``call_kwargs``.

    Returns:
        Human-readable notes (export probe, partition counts, build time) for the report.
    """
    import torch_tensorrt  # type: ignore[import-not-found]

    notes: list[str] = []
    transformer = pipe.transformer
    captured = capture_transformer_inputs(pipe, call_kwargs)
    if pin_positional_embeddings(transformer, tuple(captured["hidden_states"].shape)):
        notes.append("pinned sin-cos positional embedding for non-native resolution")
    args, has_rotary = export_inputs(captured)
    wrapper = TransformerExportWrapper(transformer, has_rotary).eval()
    probe, ep = probe_export("transformer", wrapper, args)
    notes.append(probe.summary())
    if ep is None:
        raise RuntimeError(probe.summary())
    t0 = time.perf_counter()
    compiled = torch_tensorrt.dynamo.compile(
        ep,
        arg_inputs=list(args),
        enabled_precisions={dtype},
        use_explicit_typing=False,
        truncate_double=True,
        min_block_size=1,
        require_full_compilation=False,
    )
    n_trt, n_fallback = count_trt_partitions(compiled)
    notes.append(
        f"TRT build {time.perf_counter() - t0:.0f}s; {n_trt} TensorRT engine(s), {n_fallback} PyTorch fallback subgraph(s)"
    )
    pipe.transformer = TRTTransformerShim(
        compiled,
        transformer.config,
        transformer.dtype,
        transformer.device,
        tuple(a.shape for a in args),
        has_rotary,
    )
    return notes
