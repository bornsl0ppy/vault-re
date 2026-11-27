import types

import numpy as np
import pytest

from coins_read import by_day, holding, mixed

DAY, HOUR = 86_400_000, 3_600_000


def test_holding_btc_earns_each_day_from_one_midnight_close_to_the_next():
    prices = np.array([100.0, 110.0, 99.0])
    btc = types.SimpleNamespace(first_open=0, close=np.repeat(prices, 24))
    days = np.arange(1, 4) * DAY
    assert holding({"BTCUSDT": btc}, days) == pytest.approx([0.1, -0.1])


def test_by_day_averages_the_coins_present_each_day():
    days, values = by_day([{1: 0.1, 2: 0.3}, {2: 0.1, 3: -0.2}])
    assert days.tolist() == [1, 2, 3]
    assert values == pytest.approx([0.1, 0.2, -0.2])


def test_a_mix_with_an_uncorrelated_equal_sharpe_series_lifts_the_sharpe():
    rng = np.random.default_rng(0)
    a, b = rng.normal(0.001, 0.01, 5000), rng.normal(0.002, 0.02, 5000)
    sharpe = lambda r: r.mean() / r.std() * np.sqrt(365)
    assert mixed(b, a) > sharpe(a)
