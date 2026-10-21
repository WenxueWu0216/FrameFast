"""Run the optimization ladder and write ``results.json`` + the Markdown report.

Examples::

    python -m bench.run_bench --config configs/gpu_default.toml          # full ladder on a GPU
    python -m bench.run_bench --config configs/gpu_default.toml --variants baseline,compile
    python -m bench.run_bench --config configs/cpu_smoke.toml            # CI smoke test
    python -m bench.run_bench --placeholder                              # regenerate docs/results.md stub
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from framefast.config import BenchConfig
from framefast.report import placeholder, render
from framefast.runner import BenchRunner, write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/gpu_default.toml", help="TOML config file")
    ap.add_argument("--variants", help="comma-separated variant specs (overrides the config list)")
    ap.add_argument("--runs", type=int, help="timed runs per variant")
    ap.add_argument("--warmup", type=int, help="warmup runs per variant")
    ap.add_argument("--steps", type=int, dest="num_inference_steps", help="denoising steps")
    ap.add_argument("--out-md", dest="results_md", help="Markdown report path")
    ap.add_argument("--output-dir", help="directory for results.json / videos")
    ap.add_argument("--save-videos", action="store_true", default=None, help="write mp4s for inspection")
    ap.add_argument(
        "--placeholder", action="store_true", help="only write the 'not yet measured' report and exit"
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides = {
        "runs": args.runs,
        "warmup": args.warmup,
        "num_inference_steps": args.num_inference_steps,
        "results_md": args.results_md,
        "output_dir": args.output_dir,
        "save_videos": args.save_videos,
    }
    if args.variants:
        overrides["variants"] = [v.strip() for v in args.variants.split(",") if v.strip()]
    cfg = BenchConfig.from_toml(args.config, **overrides)
    md_path = Path(cfg.results_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    if args.placeholder:
        md_path.write_text(placeholder(cfg))
        print(f"wrote placeholder report to {md_path}")
        return 0

    runner = BenchRunner(cfg)
    data = runner.run_all()
    json_path = Path(cfg.output_dir) / "results.json"
    write_json(data, json_path)
    md_path.write_text(render(data))
    print(f"wrote {json_path} and {md_path}")
    ok = [r for r in data["results"] if r["status"] == "ok"]
    failed = [r for r in data["results"] if r["status"] == "failed"]
    print(f"{len(ok)} ok, {len(failed)} failed, {len(data['results']) - len(ok) - len(failed)} skipped")
    # Baseline failing means nothing was measured: signal that to CI.
    return 0 if any(r["spec"] == "baseline" for r in ok) else 1


if __name__ == "__main__":
    sys.exit(main())
