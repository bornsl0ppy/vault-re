import numpy as np
import pytest

from backtest import simulate


def test_holding_one_times_leverage_compounds_to_the_price_path():
    moves = np.array([0.1, -0.05, 0.02])
    returns, _ = simulate(np.ones(3), moves, np.zeros(3), fee=0.0, step=1.0)
    assert np.prod(1 + returns) == pytest.approx(np.prod(1 + moves))


def test_fees_charge_the_leverage_traded_including_the_drift_back_to_the_aim():
    returns, _ = simulate(np.array([0.5, 0.5]), np.array([0.1, 0.0]), np.zeros(2), fee=0.001, step=1.0)
    assert returns[0] == pytest.approx(0.5 * 0.1 - 0.001 * 0.5)
    drifted = 0.5 * 1.1 / (1 + returns[0])
    assert returns[1] == pytest.approx(-0.001 * (drifted - 0.5))


def test_a_long_pays_positive_funding_and_a_short_collects_it():
    long_side, _ = simulate(np.array([1.0]), np.zeros(1), np.array([0.0003]), fee=0.0, step=1.0)
    short_side, _ = simulate(np.array([-1.0]), np.zeros(1), np.array([0.0003]), fee=0.0, step=1.0)
    assert long_side[0] == pytest.approx(-0.0003)
    assert short_side[0] == pytest.approx(0.0003)


def test_each_night_moves_the_step_of_the_way_to_the_aim():
    _, leverage = simulate(np.array([1.0, 1.0]), np.zeros(2), np.zeros(2), fee=0.0, step=0.8)
    assert leverage == pytest.approx([0.8, 0.96])


def test_a_loss_past_equity_wipes_the_book_and_it_stays_flat():
    returns, leverage = simulate(np.array([2.0, 2.0]), np.array([-0.6, 0.5]), np.zeros(2), fee=0.0, step=1.0)
    assert returns.tolist() == [-1.0, 0.0]
    assert leverage.tolist() == [0.0, 0.0]
