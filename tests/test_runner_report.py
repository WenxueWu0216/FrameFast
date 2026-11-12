import json
import re

from framefast.config import BenchConfig
from framefast.report import NOT_MEASURED, placeholder, render
from framefast.runner import BenchRunner, write_json


def _tiny_cfg(tmp_path, variants):
    return BenchConfig(
        name="test",
        model_id="tiny-random",
        device="cpu",
        dtype="float32",
        height=32,
        width=32,
        num_frames=9,
        num_inference_steps=2,
        warmup=1,
        runs=2,
        max_sequence_length=16,
        prompts=["a dog running", "a cat at night"],
        seeds=[0],
        variants=variants,
        output_dir=str(tmp_path),
        results_md=str(tmp_path / "results.md"),
    )


def test_runner_end_to_end(tmp_path):
    cfg = _tiny_cfg(tmp_path, ["int8_wo_ref", "bs2", "attn_flash"])
    data = BenchRunner(cfg, log=lambda *_: None).run_all()
    by = {r["spec"]: r for r in data["results"]}
    assert next(iter(by)) == "baseline"  # baseline always first
    base = by["baseline"]
    assert base["status"] == "ok" and base["latency_s"]["n"] == 2
    assert base["quality"]["psnr_db"] == "inf"  # tiny CPU pipeline is deterministic
    assert set(base["stages_ms"]) == {"text_encode", "denoise", "vae_decode", "other"}
    assert by["int8_wo_ref"]["status"] == "ok" and by["int8_wo_ref"]["quality"]["n_videos"] == 2
    assert by["bs2"]["batch_size"] == 2
    # first video of a batch uses the same initial noise as the batch-1 baseline
    assert by["bs2"]["quality"]["psnr_db"] == "inf" or float(by["bs2"]["quality"]["psnr_db"]) > 60
    if by["attn_flash"]["status"] == "skipped":
        assert "CUDA" in by["attn_flash"]["reason"]
    write_json(data, tmp_path / "results.json")
    md = render(json.loads((tmp_path / "results.json").read_text()))
    assert NOT_MEASURED in md and "not** performance results" in md
    # no speedup is ever computed for CPU runs
    assert not re.search(r"\d\.\d\dx", md)


def test_runner_records_failures(tmp_path, monkeypatch):
    from framefast import variants

    def boom(pipe, ctx):
        raise RuntimeError("exploded on purpose")

    monkeypatch.setattr(variants.get_variant("fuse_qkv"), "apply", boom)
    runner = BenchRunner(_tiny_cfg(tmp_path, ["fuse_qkv"]), log=lambda *_: None)
    data = runner.run_all()
    r = data["results"][1]
    assert r["status"] == "failed" and "exploded on purpose" in r["reason"]


def test_placeholder_report_has_no_numbers():
    md = placeholder(BenchConfig())
    assert NOT_MEASURED in md
    assert "`compile+int8_wo`" in md and "not run" in md
    assert not re.search(r"\d\.\d\dx", md)


def test_gpu_report_computes_speedup():
    data = {
        "config": BenchConfig().to_dict(),
        "env": {"device": "cuda", "gpu": "FakeGPU"},
        "lpips_available": True,
        "results": [
            {"spec": "baseline", "rung": "0", "status": "ok", "latency_s": {"p50": 10.0, "p95": 11.0}},
            {"spec": "compile", "rung": "1", "status": "ok", "latency_s": {"p50": 5.0, "p95": 5.5}},
        ],
    }
    md = render(data)
    assert NOT_MEASURED not in md
    assert "| 2.00x |" in md and "1.00x (ref)" in md
