import math

import numpy as np
import pytest

from framefast.stats import LatencySummary, percentile


@pytest.mark.parametrize("q", [0, 5, 50, 90, 95, 100])
def test_percentile_matches_numpy(q):
    rng = np.random.default_rng(0)
    xs = rng.random(17).tolist()
    assert percentile(xs, q) == pytest.approx(float(np.percentile(xs, q)))


def test_percentile_errors():
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_summary():
    s = LatencySummary.from_samples([1.0, 2.0, 3.0, 4.0])
    assert s.n == 4 and s.p50 == 2.5 and s.min == 1.0 and s.max == 4.0
    assert s.std == pytest.approx(float(np.std([1, 2, 3, 4], ddof=1)))
    assert LatencySummary.from_samples([5.0]).std == 0.0
    assert not math.isnan(s.p95)
