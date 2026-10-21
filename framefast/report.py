"""Render ``results.json`` into the Markdown report (``docs/results.md``).

The report is the *only* place performance numbers appear, and they are always produced by
the harness. A results file not measured on a CUDA device is labelled
``NOT YET MEASURED ON GPU`` in the header, and no speedup column is computed for it.
"""

from __future__ import annotations

import math
from typing import Any

from framefast.config import DEFAULT_LADDER, BenchConfig
from framefast.variants import RUNGS, resolve

NOT_MEASURED = "NOT YET MEASURED ON GPU"


def _f(x: Any, fmt: str = "{:.2f}") -> str:
    if x is None:
        return "—"
    if isinstance(x, str):
        return "∞" if x == "inf" else x
    if isinstance(x, float) and math.isinf(x):
        return "∞"
    return fmt.format(x)


def _measured_on_gpu(data: dict[str, Any]) -> bool:
    return str(data.get("env", {}).get("device", "")).startswith("cuda") and any(
        r.get("status") == "ok" for r in data.get("results", [])
    )


def render(data: dict[str, Any]) -> str:
    """Render a results dict (as produced by :meth:`BenchRunner.run_all`) to Markdown."""
    cfg = data["config"]
    env = data.get("env", {})
    on_gpu = _measured_on_gpu(data)
    is_tiny = cfg.get("model_id") == "tiny-random"
    lines: list[str] = ["# FrameFast benchmark results", ""]
    if not on_gpu:
        lines += [f"> **{NOT_MEASURED}.**", ">"]
        if is_tiny:
            lines += [
                "> This file comes from the CPU smoke configuration (tiny random-weight model). It only",
                "> proves the harness runs end to end; its timings are **not** performance results.",
            ]
        else:
            lines += ["> Run `make bench` (or `scripts/run_on_gpu.sh`) on a CUDA machine to fill this table."]
        lines.append("")
    lines += [
        "## Setup",
        "",
        f"- Model: `{cfg['model_id']}` · dtype `{cfg['dtype']}` · device `{env.get('device', cfg['device'])}`",
        f"- Video: {cfg['num_frames']} frames @ {cfg['height']}x{cfg['width']}, "
        f"{cfg['num_inference_steps']} steps, CFG {cfg['guidance_scale']}",
        f"- Timing: {cfg['warmup']} warmup + {cfg['runs']} timed runs per variant "
        "(CUDA events; p95 is linearly interpolated, so with few runs it approaches the max)",
        f"- Quality set: {len(cfg['prompts'])} prompt(s) x {len(cfg['seeds'])} seed(s), compared with baseline outputs",
    ]
    if env.get("gpu"):
        lines.append(
            f"- GPU: {env['gpu']} ({env.get('gpu_mem_gb')} GB, CC {env.get('compute_capability')}), "
            f"driver {env.get('driver')}"
        )
    software = ", ".join(
        f"{k} {env[k]}"
        for k in ("torch", "cuda_runtime", "diffusers", "triton", "torchao", "torch_tensorrt")
        if env.get(k)
    )
    if software:
        lines.append(f"- Software: {software}")
    if env.get("git_sha") or env.get("timestamp_utc"):
        lines.append(f"- Commit `{env.get('git_sha')}` · {env.get('timestamp_utc')}")
    if not data.get("lpips_available", False):
        lines.append(f"- LPIPS unavailable ({data.get('lpips_error') or 'not computed'})")
    lines.append("")

    results = data.get("results", [])
    base = next((r for r in results if r["spec"] == "baseline" and r["status"] == "ok"), None)
    base_p50 = base["latency_s"]["p50"] if base else None

    lines += [
        "## Results",
        "",
        "| Rung | Variant | Status | p50 (s) | p95 (s) | Speedup vs baseline (p50) | Frames/s | Peak GB | "
        "Cold start (s) | PSNR (dB) | SSIM | LPIPS |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lat = r.get("latency_s") or {}
        q = r.get("quality") or {}
        speed = "—"
        if on_gpu and base_p50 and r["status"] == "ok" and lat:
            speed = "1.00x (ref)" if r["spec"] == "baseline" else f"{base_p50 / lat['p50']:.2f}x"
        status = r["status"] + (f": {r['reason']}" if r.get("reason") else "")
        lines.append(
            f"| {r['rung']} {RUNGS.get(r['rung'], '')} | `{r['spec']}` | {status} | {_f(lat.get('p50'))} | "
            f"{_f(lat.get('p95'))} | {speed} | {_f(r.get('frames_per_s'))} | {_f(r.get('peak_mem_gb'))} | "
            f"{_f(r.get('cold_start_s'), '{:.1f}')} | {_f(q.get('psnr_db'))} | {_f(q.get('ssim'), '{:.4f}')} | "
            f"{_f(q.get('lpips'), '{:.4f}')} |"
        )
    lines += [
        "",
        "Batch variants (`bsN`) report latency per *call* (N videos); compare them on Frames/s. "
        "The baseline quality row compares the baseline with a second baseline run (noise floor).",
        "",
        "## Per-stage breakdown (median ms per call)",
        "",
        "| Variant | Text encode | Denoise (all steps) | Per step | VAE decode | Other |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        st = r.get("stages_ms")
        if not st:
            continue
        lines.append(
            f"| `{r['spec']}` | {_f(st['text_encode'], '{:.1f}')} | {_f(st['denoise'], '{:.1f}')} | "
            f"{_f(r.get('step_ms_median'), '{:.1f}')} | {_f(st['vae_decode'], '{:.1f}')} | {_f(st['other'], '{:.1f}')} |"
        )
    lines += ["", "## Notes per variant", ""]
    for r in results:
        notes = list(r.get("notes") or [])
        if r.get("graph_breaks"):
            notes.append(f"dynamo graph breaks during warmup: {r['graph_breaks']}")
        if notes:
            lines.append(f"- `{r['spec']}`: " + "; ".join(notes))
    lines.append("")
    lines.append("_Generated by `python -m bench.run_bench` — do not edit by hand._")
    lines.append("")
    return "\n".join(lines)


def placeholder(cfg: BenchConfig | None = None) -> str:
    """Report for a ladder that has not been measured yet (committed as ``docs/results.md``)."""
    cfg = cfg or BenchConfig()
    results = []
    for spec in cfg.variants or list(DEFAULT_LADDER):
        rv = resolve(spec)
        results.append(
            {"spec": spec, "rung": rv.rung, "description": rv.description, "status": "not run", "notes": []}
        )
    data = {
        "config": cfg.to_dict(),
        "env": {"device": "not measured"},
        "lpips_available": False,
        "lpips_error": "not run",
        "results": results,
    }
    return render(data)
