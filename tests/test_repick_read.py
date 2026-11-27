import numpy as np
import pytest

from backtest import simulate
from repick_read import pick, trade


def ranked(*coins):
    return list(coins)


def test_the_first_month_takes_the_top_ten():
    ranking = [f"C{i}" for i in range(30)]
    assert pick(ranking, set()) == {f"C{i}" for i in range(10)}


def test_a_member_keeps_its_slot_inside_the_top_fifteen_and_loses_it_below():
    ranking = [f"N{i}" for i in range(11)] + ["KEPT", "x1", "x2", "x3", "x4", "DROPPED"]
    current = {"KEPT", "DROPPED"} | {f"N{i}" for i in range(8)}
    picked = pick(ranking, current)
    assert "KEPT" in picked                          # rank 12
    assert "DROPPED" not in picked                   # rank 17
    assert len(picked) == 10
    assert "N8" in picked and "N9" not in picked     # the free slot goes to the best newcomer


def test_a_member_that_stopped_trading_drops_out():
    ranking = [f"C{i}" for i in range(20)]
    current = {f"C{i}" for i in range(9)} | {"DEAD"}
    picked = pick(ranking, current)
    assert "DEAD" not in picked and len(picked) == 10


def test_an_always_member_coin_trades_exactly_like_the_engine():
    rng = np.random.default_rng(3)
    aim, moves, funding = rng.normal(0, 1, 300), rng.normal(0, 0.03, 300), rng.normal(0, 0.0003, 300)
    ours, our_leverage = trade(aim, moves, funding, 0.0005, np.ones(300, bool))
    theirs, their_leverage = simulate(aim, moves, funding, 0.0005)
    assert ours == pytest.approx(theirs)
    assert our_leverage == pytest.approx(their_leverage)


def test_off_the_list_a_position_holds_when_the_rule_wants_more():
    aim = np.array([1.0, 2.0])
    returns, leverage = trade(aim, np.zeros(2), np.zeros(2), 0.0, np.array([True, False]))
    assert leverage.tolist() == pytest.approx([0.8, 0.8])


def test_off_the_list_a_position_shrinks_with_the_rule():
    aim = np.array([1.0, 0.5])
    _, leverage = trade(aim, np.zeros(2), np.zeros(2), 0.0, np.array([True, False]))
    assert leverage[1] == pytest.approx(0.8 + 0.8 * (0.5 - 0.8))


def test_off_the_list_a_position_closes_when_the_aim_flips_and_stays_out():
    aim = np.array([1.0, -1.0, 1.0])
    returns, leverage = trade(aim, np.zeros(3), np.zeros(3), 0.001, np.array([True, False, False]))
    assert leverage.tolist() == [0.8, 0.0, 0.0]
    assert returns[1] == pytest.approx(-0.001 * 0.8 / (1 - 0.001 * 0.8))     # the entry fee drifted leverage up
    assert returns[2] == 0.0


def test_a_returning_member_starts_from_flat():
    aim = np.array([1.0, -1.0, 1.0])
    _, leverage = trade(aim, np.zeros(3), np.zeros(3), 0.0, np.array([True, False, True]))
    assert leverage[2] == pytest.approx(0.8)
