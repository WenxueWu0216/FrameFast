import math

import pytest
import torch

from framefast.quality import LPIPSMetric, compare_videos, psnr, ssim


def _video(seed: int, f: int = 3, h: int = 32, w: int = 40) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(f, 3, h, w, generator=g)


def test_psnr_identical_and_known_value():
    v = _video(0)
    assert math.isinf(psnr(v, v))
    noisy = (v + 0.1).clamp(max=10)  # constant offset of 0.1 -> MSE 0.01 -> 20 dB
    assert psnr(v, noisy) == pytest.approx(20.0, abs=1e-6)


def test_ssim_matches_skimage():
    skm = pytest.importorskip("skimage.metrics")
    a = _video(1)
    b = (a + 0.05 * torch.randn(a.shape, generator=torch.Generator().manual_seed(2))).clamp(0, 1)
    ours = ssim(a, b)
    ref = 0.0
    for i in range(a.shape[0]):
        ref += skm.structural_similarity(
            a[i].permute(1, 2, 0).numpy().astype("float64"),
            b[i].permute(1, 2, 0).numpy().astype("float64"),
            channel_axis=-1,
            data_range=1.0,
            gaussian_weights=True,
            sigma=1.5,
            use_sample_covariance=False,
        )
    assert ours == pytest.approx(ref / a.shape[0], abs=1e-4)


def test_ssim_identical_is_one():
    v = _video(3)
    assert ssim(v, v) == pytest.approx(1.0, abs=1e-9)


def test_compare_videos_aggregation():
    a, b = _video(4), _video(5)
    rep = compare_videos([a, a], [a, (a + 0.1).clamp(max=10)])
    # one identical video (MSE 0) + one at 20 dB (MSE .01) -> mean MSE .005 -> ~23.01 dB
    assert rep.psnr_db == pytest.approx(10 * math.log10(1 / 0.005))
    assert rep.min_psnr_db == pytest.approx(20.0)
    assert rep.n_videos == 2 and rep.lpips is None
    assert compare_videos([b], [b]).as_dict()["psnr_db"] == "inf"
    with pytest.raises(ValueError):
        compare_videos([a], [])


def test_shape_mismatch():
    with pytest.raises(ValueError):
        psnr(_video(0), _video(0, f=2))


def test_lpips_degrades_gracefully():
    m = LPIPSMetric("cpu")
    if not m.available:
        assert m(_video(0), _video(1)) is None and m.error
    else:  # weights present (e.g. a GPU box): identical inputs score ~0
        assert m(_video(0), _video(0)) == pytest.approx(0.0, abs=1e-5)
