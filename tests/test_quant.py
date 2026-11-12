import pytest
import torch
import torch.nn as nn

from framefast.quant import (
    Int8WeightOnlyLinear,
    apply_reference_int8,
    module_bytes,
    quantize_per_channel_int8,
)


def test_per_channel_int8_roundtrip_error_bound():
    g = torch.Generator().manual_seed(0)
    w = torch.randn(64, 128, generator=g) * torch.linspace(0.01, 3, 64)[:, None]
    q, s = quantize_per_channel_int8(w)
    assert q.dtype == torch.int8 and s.shape == (64, 1)
    assert q.abs().max() <= 127
    # rounding error is at most half a quantization step per element
    assert ((q.float() * s - w).abs() <= s / 2 + 1e-7).all()


def test_int8_linear_matches_float_linear():
    torch.manual_seed(0)
    lin = nn.Linear(256, 512)
    qlin = Int8WeightOnlyLinear(lin)
    x = torch.randn(8, 33, 256)
    ref, out = lin(x), qlin(x)
    rel = (out - ref).norm() / ref.norm()
    assert rel < 1e-2, rel
    assert qlin.weight.shape == lin.weight.shape


def test_apply_reference_int8_on_tiny_transformer(tiny_pipe):
    tr = tiny_pipe.transformer.eval()
    g = torch.Generator().manual_seed(0)
    kw = dict(
        hidden_states=torch.randn(2, 3, 4, 4, 4, generator=g),
        encoder_hidden_states=torch.randn(2, 16, 32, generator=g),
        timestep=torch.tensor([500, 500]),
        return_dict=False,
    )
    before = module_bytes(tr)
    with torch.no_grad():
        ref = tr(**kw)[0]
        n = apply_reference_int8(tr)
        out = tr(**kw)[0]
    assert n > 0
    assert sum(isinstance(m, Int8WeightOnlyLinear) for m in tr.modules()) == n
    assert module_bytes(tr) < before
    rel = ((out - ref).norm() / ref.norm()).item()
    assert rel < 5e-2, rel


def test_torchao_int8_on_cpu_if_available():
    pytest.importorskip("torchao")
    from framefast.quant import apply_torchao

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    x = torch.randn(4, 64)
    ref = model(x)
    try:
        apply_torchao(model, "int8_wo")
    except Exception as exc:  # torchao CPU support varies by version
        pytest.skip(f"torchao int8_wo unavailable on CPU: {exc}")
    out = model(x)
    assert ((out - ref).norm() / ref.norm()) < 2e-2
