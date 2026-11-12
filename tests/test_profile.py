from bench.profile import categorize, main


def test_categorize():
    assert categorize("aten::native_layer_norm") == "normalization"
    assert categorize("aten::addmm") == "gemm"
    assert categorize("aten::_scaled_dot_product_flash_attention") == "attention"
    assert categorize("aten::mul") == "elementwise / copy"
    assert categorize("aten::cudnn_convolution") == "conv"
    assert categorize("aten::view") == "other"


def test_profile_cli_on_tiny_model(tmp_path):
    out = tmp_path / "profile.md"
    assert main(["--config", "configs/cpu_smoke.toml", "--calls", "1", "--out", str(out)]) == 0
    md = out.read_text()
    assert "NOT YET MEASURED ON GPU" in md
    assert "`CogVideoXLayerNormZero`" in md and "aten::native_layer_norm" in md
