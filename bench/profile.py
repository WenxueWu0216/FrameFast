"""Profile the denoising transformer (and VAE decode) with ``torch.profiler``.

Writes a Markdown report with

1. a **module-level breakdown**: share of transformer device time spent inside each module
   class (``CogVideoXLayerNormZero``, ``Attention``, ``FeedForward``, ...), measured with
   ``record_function`` ranges opened/closed by forward hooks;
2. an **op-category breakdown** (GEMM / attention / normalization / elementwise / conv / other)
   from the ``aten::`` ops' *self* device time (no double counting of children);
3. the **top-K aten ops** by self device time, for the transformer and for the VAE decode;
4. a Chrome trace (``--trace``) for drill-down in Perfetto / chrome://tracing.

This report is the evidence used to decide which op deserves a custom kernel: the fused
LayerNorm+modulation kernel (``framefast/kernels/adaln.py``) targets the
``CogVideoXLayerNormZero`` row and the normalization + elementwise categories.

Example::

    python -m bench.profile --config configs/gpu_default.toml --out docs/profile_top_ops.md
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from framefast.config import BenchConfig
from framefast.report import NOT_MEASURED
from framefast.runner import BenchRunner, environment_info
from framefast.trt import capture_transformer_inputs
from framefast.variants import RunContext, resolve

MODULE_CLASSES = (
    "CogVideoXLayerNormZero",
    "Attention",
    "FeedForward",
    "CogVideoXPatchEmbed",
    "TimestepEmbedding",
    "AdaLayerNorm",
    "LayerNorm",
)

CATEGORIES: list[tuple[str, re.Pattern[str]]] = [
    ("attention", re.compile(r"flash|attention|fmha|sdpa|softmax|bmm|baddbmm", re.I)),
    ("gemm", re.compile(r"aten::(mm|addmm|matmul|linear|_scaled_mm|_int_mm)|gemm|cutlass|cublas", re.I)),
    ("normalization", re.compile(r"layer_norm|group_norm|rms_norm|batch_norm", re.I)),
    ("conv", re.compile(r"conv", re.I)),
    (
        "elementwise / copy",
        re.compile(r"aten::(add|mul|sub|div|silu|gelu|tanh|sigmoid|exp|cos|sin|where|copy_|cat|clone|"
                   r"contiguous|_to_copy|to|neg|pow|rsqrt|fill_|zero_|lerp|clamp|cumsum|baddbmm|addcmul)", re.I),
    ),
]  # fmt: skip


def categorize(op_name: str) -> str:
    """Map an op/kernel name to a coarse category."""
    for cat, pat in CATEGORIES:
        if pat.search(op_name):
            return cat
    return "other"


def _self_time_us(evt: Any, cuda: bool) -> float:
    if cuda:
        return float(getattr(evt, "self_device_time_total", getattr(evt, "self_cuda_time_total", 0.0)))
    return float(evt.self_cpu_time_total)


def _total_time_us(evt: Any, cuda: bool) -> float:
    if cuda:
        return float(getattr(evt, "device_time_total", getattr(evt, "cuda_time_total", 0.0)))
    return float(evt.cpu_time_total)


def attach_module_ranges(model: torch.nn.Module, class_names: tuple[str, ...]) -> Callable[[], None]:
    """Open a ``record_function("module::<Class>")`` range around every matching submodule.

    Returns:
        A function that removes the hooks.
    """
    handles = []
    stacks: dict[int, list[Any]] = defaultdict(list)

    def pre(mod: torch.nn.Module, args: Any) -> None:
        rf = record_function(f"module::{type(mod).__name__}")
        rf.__enter__()
        stacks[id(mod)].append(rf)

    def post(mod: torch.nn.Module, args: Any, out: Any) -> None:
        stacks[id(mod)].pop().__exit__(None, None, None)

    for m in model.modules():
        if type(m).__name__ in class_names:
            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(post))

    def remove() -> None:
        for h in handles:
            h.remove()

    return remove


def summarize(prof: Any, cuda: bool, top_k: int, scope: str) -> dict[str, Any]:
    """Extract module shares, category shares and top ops from a finished profiler."""
    events = prof.key_averages()
    scope_total = 0.0
    modules: dict[str, float] = {}
    ops: list[tuple[str, float, int]] = []
    for e in events:
        if e.key == scope:
            scope_total = _total_time_us(e, cuda)
        elif e.key.startswith("module::"):
            modules[e.key[len("module::") :]] = _total_time_us(e, cuda)
        elif e.key.startswith("aten::"):
            t = _self_time_us(e, cuda)
            if t > 0:
                ops.append((e.key, t, e.count))
    ops.sort(key=lambda x: -x[1])
    op_total = sum(t for _, t, _ in ops) or 1.0
    cats: dict[str, float] = defaultdict(float)
    for name, t, _ in ops:
        cats[categorize(name)] += t
    return {
        "scope_total_us": scope_total,
        "modules": modules,
        "categories": dict(sorted(cats.items(), key=lambda kv: -kv[1])),
        "op_total_us": op_total,
        "top_ops": ops[:top_k],
    }


def render_md(cfg: BenchConfig, env: dict[str, Any], tr: dict[str, Any], vae: dict[str, Any] | None,
              cuda: bool, n_calls: int, variant: str) -> str:  # fmt: skip
    """Markdown for the profile report."""
    unit = "device (GPU kernel) time" if cuda else "CPU time"
    lines = ["# FrameFast profile: top ops", ""]
    if not cuda:
        lines += [
            f"> **{NOT_MEASURED}.** This profile was taken on CPU"
            + (" with the tiny random-weight model" if cfg.model_id == "tiny-random" else "")
            + "; op shares on CPU say nothing about GPU bottlenecks.",
            "",
        ]
    lines += [
        f"- Model `{cfg.model_id}`, variant `{variant}`, {cfg.num_frames} frames @ {cfg.height}x{cfg.width}, "
        f"CFG {cfg.guidance_scale}; {n_calls} profiled transformer call(s) after warmup.",
        f"- Times are {unit}; shares are relative to the profiled transformer calls.",
        f"- {env.get('gpu', env.get('device'))} · torch {env.get('torch')} · commit `{env.get('git_sha')}`",
        "",
        "## Transformer: time inside module classes",
        "",
        "| Module class | Total time (ms, all calls) | Share of transformer |",
        "|---|---:|---:|",
    ]
    total = tr["scope_total_us"] or tr["op_total_us"]
    for name, t in sorted(tr["modules"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| `{name}` | {t / 1e3:.2f} | {100 * t / total:.1f}% |")
    lines += [
        "",
        "Rows can nest, so shares do not sum to 100%: `LayerNorm` counts every `nn.LayerNorm`",
        "(including the one inside each `CogVideoXLayerNormZero` and the Q/K norms inside `Attention`).",
        "`Attention` includes the QKV/out projections; `CogVideoXLayerNormZero` includes its small",
        "conditioning GEMM (`linear(silu(temb))`) plus the LayerNorm + modulation chain.",
        "",
        "## Transformer: op categories (self time of aten ops)",
        "",
        "| Category | Time (ms) | Share |",
        "|---|---:|---:|",
    ]
    for cat, t in tr["categories"].items():
        lines.append(f"| {cat} | {t / 1e3:.2f} | {100 * t / tr['op_total_us']:.1f}% |")

    def top_table(title: str, s: dict[str, Any]) -> None:
        lines.extend(["", f"## {title}", "", "| # | Op | Calls | Self time (ms) | Share | Category |",
                      "|---:|---|---:|---:|---:|---|"])  # fmt: skip
        for i, (name, t, count) in enumerate(s["top_ops"], 1):
            share = 100 * t / s["op_total_us"]
            lines.append(f"| {i} | `{name}` | {count} | {t / 1e3:.2f} | {share:.1f}% | {categorize(name)} |")

    top_table("Transformer: top aten ops by self time", tr)
    if vae is not None:
        top_table("VAE decode: top aten ops by self time", vae)
    lines += ["", "_Generated by `python -m bench.profile` — do not edit by hand._", ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/gpu_default.toml")
    ap.add_argument(
        "--variant",
        default="baseline",
        help="variant spec to profile (module/aten tables are meaningful for eager variants; "
        "inductor-generated kernels of compiled variants only show up in the trace)",
    )
    ap.add_argument("--calls", type=int, default=3, help="profiled transformer forward calls")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--no-vae", action="store_true", help="skip profiling the VAE decode")
    ap.add_argument("--out", default="docs/profile_top_ops.md")
    ap.add_argument("--trace", default=None, help="write a Chrome trace JSON here")
    args = ap.parse_args(argv)

    cfg = BenchConfig.from_toml(args.config)
    runner = BenchRunner(cfg, log=lambda *_: None)
    cuda = runner.cuda
    rv = resolve(args.variant)
    if (reason := rv.skip_reason()) is not None:
        print(f"cannot profile {args.variant}: {reason}")
        return 1
    pipe = runner._load(rv)
    prompt, seed = cfg.quality_pairs[0]
    call = runner.call_kwargs(prompt, seed, rv.batch_size)
    with torch.no_grad():
        rv.apply(pipe, RunContext(runner.device, runner.dtype, dict(call)))
    with torch.no_grad(), rv.call_context():
        pipe(**dict(call, generator=torch.Generator().manual_seed(seed)))  # warmup / compile
        inputs = capture_transformer_inputs(pipe, dict(call, generator=torch.Generator().manual_seed(seed)))
        transformer = pipe.transformer
        labels = "compile" not in args.variant  # hooks would cause graph breaks under torch.compile
        remove = attach_module_ranges(transformer, MODULE_CLASSES) if labels else (lambda: None)
        activities = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
        transformer(**inputs)  # warm the exact call once more outside the pipeline
        if cuda:
            torch.cuda.synchronize()
        with profile(activities=activities, record_shapes=False) as prof:
            with record_function("framefast::transformer"):
                for _ in range(args.calls):
                    transformer(**inputs)
            if cuda:
                torch.cuda.synchronize()
        remove()
        tr = summarize(prof, cuda, args.top, "framefast::transformer")
        if args.trace:
            Path(args.trace).parent.mkdir(parents=True, exist_ok=True)
            prof.export_chrome_trace(args.trace)

        vae_summary = None
        if not args.no_vae:
            b2, f, c, h, w = inputs["hidden_states"].shape
            batch = b2 // 2 if cfg.guidance_scale > 1 else b2
            z = torch.randn(batch, c, f, h, w, device=runner.device, dtype=pipe.vae.dtype)
            pipe.vae.decode(z)
            with profile(activities=activities) as vprof:
                with record_function("framefast::vae"):
                    pipe.vae.decode(z)
                if cuda:
                    torch.cuda.synchronize()
            vae_summary = summarize(vprof, cuda, args.top, "framefast::vae")

    md = render_md(cfg, environment_info(runner.device), tr, vae_summary, cuda, args.calls, args.variant)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
