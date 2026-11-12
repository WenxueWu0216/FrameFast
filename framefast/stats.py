"""Summary statistics for latency samples."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any


def percentile(samples: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile (same definition as ``numpy.percentile`` default).

    Args:
        samples: Non-empty sequence of values.
        q: Percentile in ``[0, 100]``.
    """
    if not samples:
        raise ValueError("percentile of empty sequence")
    if not 0.0 <= q <= 100.0:
        raise ValueError(f"q must be in [0, 100], got {q}")
    xs = sorted(samples)
    pos = (len(xs) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    if lo == hi:
        return float(xs[lo])
    return float(xs[lo] + (xs[hi] - xs[lo]) * (pos - lo))


@dataclass
class LatencySummary:
    """p50/p95/mean/std/min/max over ``n`` samples (same unit as the input)."""

    n: int
    p50: float
    p95: float
    mean: float
    std: float
    min: float
    max: float

    @classmethod
    def from_samples(cls, samples: Sequence[float]) -> LatencySummary:
        """Summarise a non-empty list of samples (std is the sample standard deviation)."""
        n = len(samples)
        mean = sum(samples) / n
        var = sum((x - mean) ** 2 for x in samples) / (n - 1) if n > 1 else 0.0
        return cls(
            n=n,
            p50=percentile(samples, 50),
            p95=percentile(samples, 95),
            mean=mean,
            std=math.sqrt(var),
            min=min(samples),
            max=max(samples),
        )

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable representation."""
        return asdict(self)


def median(samples: Sequence[float]) -> float:
    """Median of a non-empty sequence."""
    return percentile(samples, 50)
