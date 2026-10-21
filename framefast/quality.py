"""Output-quality metrics of an optimized variant against baseline outputs.

All metrics compare videos frame-by-frame. Inputs are float tensors in ``[0, 1]`` shaped
``(F, C, H, W)`` (one video) — exactly what the pipeline returns with ``output_type="pt"``.

* **PSNR** (dB, higher is better; ``inf`` for identical outputs).
* **SSIM** (Wang et al. 2004; 11x11 Gaussian window, sigma 1.5, K1=0.01, K2=0.03, computed per
  channel and averaged; validated against ``skimage.metrics.structural_similarity`` in tests).
* **LPIPS** (AlexNet, Zhang et al. 2018; lower is better). Optional — needs the ``lpips``
  package and its pretrained weights, so it is reported as ``None`` when unavailable.

Note that diffusion sampling is iterative: tiny numerical differences (a different matmul
reduction order under ``torch.compile``) can grow over the denoising trajectory, so a fast
variant can produce a plausible but *different* video. Pixel metrics then understate perceptual
quality, which is why LPIPS is reported alongside PSNR/SSIM.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F


def psnr(ref: torch.Tensor, test: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio over the whole tensor, in dB."""
    _check_pair(ref, test)
    mse = torch.mean((ref.double() - test.double()) ** 2).item()
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10(data_range**2 / mse)


def _gaussian_window(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float64, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)


def ssim(
    ref: torch.Tensor,
    test: torch.Tensor,
    data_range: float = 1.0,
    win_size: int = 11,
    sigma: float = 1.5,
) -> float:
    """Mean structural similarity over frames and channels.

    Matches ``skimage.metrics.structural_similarity(..., gaussian_weights=True, sigma=1.5,
    use_sample_covariance=False, data_range=1.0)`` (valid-region mean, no padding).

    Args:
        ref, test: ``(F, C, H, W)`` or ``(C, H, W)`` tensors in ``[0, data_range]``.
    """
    _check_pair(ref, test)
    x = ref.double()
    y = test.double()
    if x.dim() == 3:
        x, y = x[None], y[None]
    _, c, h, w = x.shape
    win = min(win_size, h, w)
    if win % 2 == 0:
        win -= 1
    kernel = _gaussian_window(win, sigma, x.device)[None, None].expand(c, 1, win, win)

    def filt(t: torch.Tensor) -> torch.Tensor:
        return F.conv2d(t, kernel, groups=c)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = filt(x), filt(y)
    sxx = filt(x * x) - mu_x**2
    syy = filt(y * y) - mu_y**2
    sxy = filt(x * y) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    return float((num / den).mean().item())


class LPIPSMetric:
    """Lazy wrapper around the ``lpips`` package (AlexNet backbone).

    ``available`` is ``False`` (and :meth:`__call__` returns ``None``) when the package or its
    weights cannot be loaded, e.g. in an offline CI container.
    """

    def __init__(self, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._net: Any = None
        self.error: str | None = None
        try:
            import lpips  # type: ignore[import-not-found]

            self._net = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()
        except Exception as exc:  # missing package, no network for weights, ...
            self.error = f"{type(exc).__name__}: {exc}"

    @property
    def available(self) -> bool:
        """Whether LPIPS can be computed."""
        return self._net is not None

    @torch.no_grad()
    def __call__(self, ref: torch.Tensor, test: torch.Tensor, batch: int = 8) -> float | None:
        """Mean LPIPS over frames of ``(F, C, H, W)`` tensors in ``[0, 1]``."""
        if self._net is None:
            return None
        _check_pair(ref, test)
        vals: list[torch.Tensor] = []
        for i in range(0, ref.shape[0], batch):
            a = ref[i : i + batch].to(self.device, torch.float32) * 2 - 1
            b = test[i : i + batch].to(self.device, torch.float32) * 2 - 1
            vals.append(self._net(a, b).flatten().cpu())
        return float(torch.cat(vals).mean().item())


@dataclass
class QualityReport:
    """Quality of one variant vs baseline, averaged over the (prompt, seed) set."""

    psnr_db: float
    ssim: float
    lpips: float | None
    n_videos: int
    min_psnr_db: float

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation (``inf`` is encoded as the string ``"inf"``)."""
        d = asdict(self)
        for k in ("psnr_db", "min_psnr_db"):
            if math.isinf(d[k]):
                d[k] = "inf"
        return d


def compare_videos(
    refs: list[torch.Tensor],
    tests: list[torch.Tensor],
    lpips_metric: LPIPSMetric | None = None,
) -> QualityReport:
    """Average PSNR/SSIM/LPIPS between paired videos.

    PSNR is averaged in the MSE domain per video then converted (so one identical video does
    not make the mean ``inf`` unless all are identical); ``min_psnr_db`` is the worst video.
    """
    if len(refs) != len(tests) or not refs:
        raise ValueError("need equally many (non-zero) reference and test videos")
    psnrs = [psnr(r, t) for r, t in zip(refs, tests, strict=True)]
    finite = [p for p in psnrs if not math.isinf(p)]
    if finite:
        mean_mse = sum(10 ** (-p / 10) for p in finite) / len(psnrs)
        mean_psnr = -10 * math.log10(mean_mse)
    else:
        mean_psnr = math.inf
    ssims = [ssim(r, t) for r, t in zip(refs, tests, strict=True)]
    lp: float | None = None
    if lpips_metric is not None and lpips_metric.available:
        vals = [lpips_metric(r, t) for r, t in zip(refs, tests, strict=True)]
        lp = sum(v for v in vals if v is not None) / len(vals)
    return QualityReport(
        psnr_db=mean_psnr,
        ssim=sum(ssims) / len(ssims),
        lpips=lp,
        n_videos=len(refs),
        min_psnr_db=min(psnrs),
    )


def _check_pair(a: torch.Tensor, b: torch.Tensor) -> None:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
