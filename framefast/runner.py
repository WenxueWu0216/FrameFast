"""Benchmark runner: measures each variant in isolation against the baseline.

For every variant spec the runner

1. loads a **fresh** pipeline (so no optimization leaks into the next measurement),
2. places it (device / model offload / sequential offload) and applies the variant,
3. runs ``warmup`` untimed calls (the first one is reported as *cold start*: it includes
   compilation, autotuning or engine building),
4. runs ``runs`` timed calls with per-stage CUDA-event timing and records peak memory,
5. generates every (prompt, seed) pair of the quality set and compares it with the baseline's
   output for the same pair (PSNR / SSIM / LPIPS).

Failures (OOM, unsupported op, export error) are caught and recorded per variant; the run
continues with the next variant.
"""

from __future__ import annotations

import contextlib
import gc
import json
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from framefast.config import BenchConfig
from framefast.pipeline import TINY_MODEL_ID, load_pipeline, resolve_dtype
from framefast.quality import LPIPSMetric, compare_videos
from framefast.stats import LatencySummary, median
from framefast.timing import StageTimer, StageTimes, wall_clock
from framefast.variants import ResolvedVariant, RunContext, resolve


@dataclass
class VariantResult:
    """Everything measured for one variant (serialised to ``results.json``)."""

    spec: str
    rung: str
    description: str
    status: str  # "ok" | "skipped" | "failed"
    reason: str | None = None
    notes: list[str] = field(default_factory=list)
    batch_size: int = 1
    cold_start_s: float | None = None
    latency_s: dict[str, Any] | None = None
    stages_ms: dict[str, float] | None = None
    step_ms_median: float | None = None
    frames_per_s: float | None = None
    peak_mem_gb: float | None = None
    peak_reserved_gb: float | None = None
    quality: dict[str, Any] | None = None
    graph_breaks: int | None = None
    runs: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def environment_info(device: torch.device) -> dict[str, Any]:
    """Versions and hardware details stored with every result file."""

    def ver(mod: str) -> str | None:
        try:
            return __import__(mod).__version__
        except Exception:
            return None

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "diffusers": ver("diffusers"),
        "transformers": ver("transformers"),
        "triton": ver("triton"),
        "torchao": ver("torchao"),
        "torch_tensorrt": ver("torch_tensorrt"),
        "device": str(device),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        info.update(
            gpu=props.name,
            gpu_mem_gb=round(props.total_memory / 1e9, 1),
            compute_capability=f"{props.major}.{props.minor}",
        )
        try:
            info["driver"] = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
            ).strip()
        except Exception:
            info["driver"] = None
    try:
        info["git_sha"] = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        info["git_sha"] = None
    return info


class BenchRunner:
    """Runs the configured variant ladder and collects :class:`VariantResult` objects."""

    def __init__(self, cfg: BenchConfig, log: Any = print) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("config asks for device='cuda' but no CUDA device is available")
        self.dtype = resolve_dtype(cfg.dtype)
        self.cuda = self.device.type == "cuda"
        self.log = log
        self.baseline_videos: dict[tuple[str, int], torch.Tensor] = {}
        self.lpips = LPIPSMetric(self.device)
        self.out_dir = Path(cfg.output_dir)

    # -- helpers ---------------------------------------------------------------------------
    def call_kwargs(self, prompt: str, seed: int, batch_size: int = 1) -> dict[str, Any]:
        """Pipeline kwargs for one generation. The latent noise is drawn from a CPU generator so
        it is bit-identical across devices and variants."""
        return dict(
            prompt=[prompt] * batch_size if batch_size > 1 else prompt,
            num_frames=self.cfg.num_frames,
            height=self.cfg.height,
            width=self.cfg.width,
            num_inference_steps=self.cfg.num_inference_steps,
            guidance_scale=self.cfg.guidance_scale,
            use_dynamic_cfg=False,
            max_sequence_length=self.cfg.max_sequence_length,
            generator=torch.Generator("cpu").manual_seed(seed),
            output_type="pt",
        )

    def _load(self, rv: ResolvedVariant) -> Any:
        pipe = load_pipeline(self.cfg.model_id, self.dtype, self.cfg.max_sequence_length)
        placement = rv.placement
        if placement == "device":
            pipe.to(self.device)
        elif placement == "model_offload":
            pipe.enable_model_cpu_offload(device=self.device)
        elif placement == "sequential_offload":
            pipe.enable_sequential_cpu_offload(device=self.device)
        else:  # pragma: no cover - validated by ResolvedVariant
            raise ValueError(placement)
        return pipe

    def _cleanup(self) -> None:
        gc.collect()
        if self.cuda:
            torch.cuda.empty_cache()
        with contextlib.suppress(Exception):
            torch._dynamo.reset()

    def _save_video(self, frames: torch.Tensor, name: str) -> None:
        try:
            from diffusers.utils import export_to_video

            path = self.out_dir / "videos" / f"{name}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            arr = frames.permute(0, 2, 3, 1).float().cpu().numpy()
            export_to_video(list(arr), str(path), fps=8)
        except Exception as exc:
            self.log(f"  (could not save video {name}: {exc})")

    # -- main entry points -----------------------------------------------------------------
    def run_all(self) -> dict[str, Any]:
        """Run every configured variant; baseline is always run first."""
        specs = list(dict.fromkeys(self.cfg.variants))
        if "baseline" in specs:
            specs.remove("baseline")
        specs.insert(0, "baseline")
        results = []
        for spec in specs:
            self.log(f"[framefast] variant {spec}")
            res = self.run_variant(spec)
            self.log(f"  -> {res.status}" + (f" ({res.reason})" if res.reason else ""))
            results.append(res.as_dict())
        return {
            "config": self.cfg.to_dict(),
            "env": environment_info(self.device),
            "lpips_available": self.lpips.available,
            "lpips_error": self.lpips.error,
            "results": results,
        }

    def run_variant(self, spec: str) -> VariantResult:
        """Measure one variant spec; never raises (errors are recorded in the result)."""
        rv = resolve(spec)
        res = VariantResult(spec, rv.rung, rv.description, status="ok", batch_size=rv.batch_size)
        if spec != "baseline" and not self.baseline_videos:
            res.status, res.reason = "skipped", "baseline did not produce reference outputs"
            return res
        if (reason := rv.skip_reason()) is not None:
            res.status, res.reason = "skipped", reason
            return res
        pipe = None
        try:
            pipe = self._load(rv)
            prompt0, seed0 = self.cfg.quality_pairs[0]
            ctx = RunContext(
                self.device,
                self.dtype,
                self.call_kwargs(prompt0, seed0, rv.batch_size),
                is_tiny=self.cfg.model_id == TINY_MODEL_ID,
            )
            with torch.no_grad():
                res.notes.extend(rv.apply(pipe, ctx))
            self._measure(pipe, rv, res)
            self._quality(pipe, rv, res)
        except Exception as exc:
            res.status = "failed"
            oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            res.reason = (
                "CUDA out of memory"
                if oom
                else f"{type(exc).__name__}: {str(exc).splitlines()[0][:300] if str(exc) else ''}"
            )
            res.notes.append(
                "traceback tail: " + " | ".join(traceback.format_exc().strip().splitlines()[-3:])
            )
        finally:
            del pipe
            self._cleanup()
        return res

    def _measure(self, pipe: Any, rv: ResolvedVariant, res: VariantResult) -> None:
        cfg = self.cfg
        prompt0, seed0 = cfg.quality_pairs[0]
        runs = min(cfg.runs, rv.max_runs) if rv.max_runs else cfg.runs
        warmup = max(1, min(cfg.warmup, rv.max_runs or cfg.warmup))
        try:
            import torch._dynamo.utils as dyn_utils

            dyn_utils.counters.clear()
        except Exception:
            dyn_utils = None
        with rv.call_context():
            for i in range(warmup):
                _, secs = wall_clock(
                    lambda: pipe(**self.call_kwargs(prompt0, seed0, rv.batch_size)), self.cuda
                )
                if i == 0:
                    res.cold_start_s = secs
            if dyn_utils is not None:
                res.graph_breaks = sum(dyn_utils.counters["graph_break"].values())
            if self.cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            timer = StageTimer(pipe, self.cuda)
            samples: list[StageTimes] = []
            with timer.attach():
                for _ in range(runs):
                    timer.call(**self.call_kwargs(prompt0, seed0, rv.batch_size))
                    samples.append(timer.result())
        if self.cuda:
            res.peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
            res.peak_reserved_gb = torch.cuda.max_memory_reserved() / 1e9
        totals_s = [s.total_ms / 1e3 for s in samples]
        res.latency_s = LatencySummary.from_samples(totals_s).as_dict()
        res.stages_ms = {
            "text_encode": median([s.text_encode_ms for s in samples]),
            "denoise": median([s.denoise_ms for s in samples]),
            "vae_decode": median([s.vae_decode_ms for s in samples]),
            "other": median([s.other_ms for s in samples]),
        }
        res.step_ms_median = median([x for s in samples for x in s.step_ms])
        res.frames_per_s = cfg.num_frames * rv.batch_size / res.latency_s["p50"]
        res.runs = [s.as_dict() for s in samples]

    def _quality(self, pipe: Any, rv: ResolvedVariant, res: VariantResult) -> None:
        videos: dict[tuple[str, int], torch.Tensor] = {}
        with rv.call_context():
            for prompt, seed in self.cfg.quality_pairs:
                frames = pipe(**self.call_kwargs(prompt, seed, rv.batch_size)).frames
                # Batch variants: video 0 uses the same initial noise as the batch-1 baseline.
                videos[(prompt, seed)] = frames[0].to("cpu", torch.float16)
        if self.cfg.save_videos:
            for i, v in enumerate(videos.values()):
                self._save_video(v, f"{res.spec.replace('+', '__')}_{i}")
        if res.spec == "baseline":
            self.baseline_videos = videos
            # Run-to-run noise floor: regenerate the first pair and compare with itself. Any
            # variant whose metrics are at this level is indistinguishable from baseline.
            k0 = self.cfg.quality_pairs[0]
            with rv.call_context():
                again = pipe(**self.call_kwargs(*k0, rv.batch_size)).frames[0].to("cpu", torch.float16)
            res.quality = compare_videos([videos[k0].float()], [again.float()], self.lpips).as_dict()
            res.notes.append("quality row = baseline vs. itself (run-to-run determinism / noise floor)")
            return
        keys = list(self.baseline_videos)
        report = compare_videos(
            [self.baseline_videos[k].float() for k in keys],
            [videos[k].float() for k in keys],
            self.lpips,
        )
        res.quality = report.as_dict()


def write_json(data: dict[str, Any], path: str | Path) -> None:
    """Write results JSON (``inf`` -> ``"inf"`` is already handled by the quality module)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2, default=str))
