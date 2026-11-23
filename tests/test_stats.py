import numpy as np
import pytest

from stats import deflated_sharpe

# the implementation stats.py was ported from, run on the same series (2026-11-20)
REFERENCE = {1: (0.17682844281426838, 771.1126122361646, 0.0, 0.6015026167108521),
            5: (0.17682844281426838, 771.1126122361646, 0.8204964017408342, 0.17454245575658256),
            250: (0.17682844281426838, 771.1126122361646, 1.9523142754552656, 0.004899256426420953)}


def persistent_fat_tailed_series():
    rng = np.random.default_rng(7)
    noise = rng.standard_t(4, size=1500) * 0.02
    series = np.empty(1500)
    series[0] = noise[0]
    for i in range(1, 1500):
        series[i] = 0.3 * series[i - 1] + noise[i] + 0.0008
    return series


@pytest.mark.parametrize("trials", REFERENCE)
def test_deflated_sharpe_matches_the_reference_implementation(trials):
    measured = deflated_sharpe(persistent_fat_tailed_series(), trials)
    sharpe, effective, luck, dsr = REFERENCE[trials]
    assert measured["sharpe"] == pytest.approx(sharpe, rel=1e-12)
    assert measured["effective"] == pytest.approx(effective, rel=1e-12)
    assert measured["luck"] == pytest.approx(luck, rel=1e-12, abs=1e-15)
    assert measured["dsr"] == pytest.approx(dsr, rel=1e-10)
