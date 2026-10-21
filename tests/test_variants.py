import pytest
import torch

from framefast.config import DEFAULT_LADDER
from framefast.variants import RunContext, get_variant, registered_names, resolve


def test_default_ladder_resolves():
    for spec in DEFAULT_LADDER:
        rv = resolve(spec)
        assert rv.parts and rv.description


def test_batch_variant_and_stacking():
    assert get_variant("bs4").batch_size == 4
    rv = resolve("vae_slicing+bs2")
    assert rv.batch_size == 2 and rv.rung == "6"
    with pytest.raises(ValueError):
        resolve("bs2+bs4")
    with pytest.raises(ValueError):
        resolve("cpu_offload_model+cpu_offload_seq")
    with pytest.raises(KeyError):
        resolve("warp_drive")


def test_compile_is_applied_last(monkeypatch):
    order: list[str] = []
    rv = resolve("compile+int8_wo_ref")
    for p in rv.parts:
        monkeypatch.setattr(p, "apply", lambda pipe, ctx, n=p.name: order.append(n) or [])
    rv.apply(None, RunContext(torch.device("cpu"), torch.float32, {}))
    assert order == ["int8_wo_ref", "compile"]


def test_cuda_only_variants_skip_without_gpu():
    if torch.cuda.is_available():
        pytest.skip("GPU present")
    for name in ("compile_cudagraphs", "attn_flash", "fp8_dyn", "trt_transformer", "cpu_offload_model"):
        assert "CUDA" in (resolve(name).skip_reason() or "")


def test_call_context_nesting():
    rv = resolve("attn_math+bs2")
    with rv.call_context():
        pass
    assert "baseline" in registered_names()
