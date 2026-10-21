"""Benchmark configuration (loaded from TOML files in ``configs/``)."""

from __future__ import annotations

import dataclasses
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info >= (3, 11):
    import tomllib
else:  # pragma: no cover - exercised only on 3.10
    import tomli as tomllib

DEFAULT_PROMPTS: tuple[str, ...] = (
    "A golden retriever running through a sunlit meadow, slow motion, cinematic lighting.",
    "Aerial drone shot of waves crashing against black volcanic rocks at sunset.",
    "A steaming cup of coffee on a wooden table, camera slowly orbiting, shallow depth of field.",
)

DEFAULT_LADDER: tuple[str, ...] = (
    "baseline",
    # 1. graph compilation
    "compile",
    "compile_cudagraphs",
    # 2. attention backends
    "attn_math",
    "attn_flash",
    "attn_efficient",
    "attn_cudnn",
    "fuse_qkv",
    # 3. quantization
    "int8_wo",
    "compile+int8_wo",
    "int8_dyn",
    "fp8_dyn",
    "compile+fp8_dyn",
    # 4. custom Triton kernel
    "triton_adaln",
    "compile+triton_adaln",
    # 5. TensorRT
    "trt_transformer",
    # 6. memory / batching
    "vae_tiling",
    "vae_slicing+bs2",
    "cpu_offload_model",
    "cpu_offload_seq",
    "bs2",
    "bs4",
)


@dataclass
class BenchConfig:
    """Everything that determines a benchmark run.

    Attributes:
        name: Human-readable name of the config (shown in the report).
        model_id: Hugging Face model id, or ``"tiny-random"`` for the offline CPU test model.
        device: ``"cuda"`` or ``"cpu"``.
        dtype: Compute dtype for the baseline (``"float16"``, ``"bfloat16"`` or ``"float32"``).
        height, width, num_frames: Output video geometry.
        num_inference_steps: Denoising steps per generation.
        guidance_scale: Classifier-free guidance scale (> 1 doubles the transformer batch).
        warmup: Untimed runs before measuring (absorbs compilation / autotuning / allocator warmup).
        runs: Timed runs per variant; p50/p95 are computed over these.
        prompts: Fixed prompt set for the quality check.
        seeds: Fixed seeds for the quality check; every (prompt, seed) pair is generated once.
        variants: Variant specs to run, e.g. ``"compile+int8_wo"`` (``+`` stacks variants).
        max_sequence_length: T5 token length (226 for CogVideoX).
        save_videos: Write mp4s of baseline/variant outputs for visual inspection.
        output_dir: Where raw JSON results, traces and videos go.
        results_md: Markdown report path written by ``make bench``.
    """

    name: str = "gpu-default"
    model_id: str = "THUDM/CogVideoX-2b"
    device: str = "cuda"
    dtype: str = "float16"
    height: int = 480
    width: int = 720
    num_frames: int = 49
    num_inference_steps: int = 20
    guidance_scale: float = 6.0
    warmup: int = 2
    runs: int = 8
    prompts: list[str] = field(default_factory=lambda: list(DEFAULT_PROMPTS))
    seeds: list[int] = field(default_factory=lambda: [0])
    variants: list[str] = field(default_factory=lambda: list(DEFAULT_LADDER))
    max_sequence_length: int = 226
    save_videos: bool = False
    output_dir: str = "runs/latest"
    results_md: str = "docs/results.md"

    @classmethod
    def from_toml(cls, path: str | Path, **overrides: Any) -> BenchConfig:
        """Load a config from a TOML file; keyword ``overrides`` win over file values."""
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        data.update({k: v for k, v in overrides.items() if v is not None})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown config keys in {path}: {sorted(unknown)}")
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dict view (stored alongside results for reproducibility)."""
        return dataclasses.asdict(self)

    @property
    def quality_pairs(self) -> list[tuple[str, int]]:
        """All (prompt, seed) pairs used for the quality comparison."""
        return [(p, s) for p in self.prompts for s in self.seeds]
