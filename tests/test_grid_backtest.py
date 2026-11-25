import pytest

from grid_backtest import CAP, DEPTHS, ENTRY, FIRST_RUNG, MINUTE, TAKE_PROFIT, UNWIND_HOURS, Costs, deal

FREE = Costs(taker=0.0, maker=0.0)
MIDNIGHT = 1_700_006_400_000


def bars(*rows):
    """Minute bars from (high, low, close), the first opening at MIDNIGHT."""
    return [(MIDNIGHT + i * MINUTE, high, low, close) for i, (high, low, close) in enumerate(rows)]


def first_rung_average():
    quantity = ENTRY / 100 + ENTRY * FIRST_RUNG / (100 * (1 - DEPTHS[0]))
    return ENTRY * (1 + FIRST_RUNG) / quantity, quantity


def test_take_profit_fills_only_once_a_minute_trades_above_it():
    target = 100 * (1 + TAKE_PROFIT)
    out = deal(bars((target - 1e-6, 99.9, 100.2), (target + 0.01, 100.1, 101.2)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", FREE)
    assert out["exit"] == "take-profit"
    assert out["end"] == MIDNIGHT + 2 * MINUTE
    assert out["rungs"] == 0
    assert out["equity"] == pytest.approx(1 + ENTRY * TAKE_PROFIT)


def test_a_rung_fills_below_its_price_and_the_take_profit_follows_the_new_average():
    rung = 100 * (1 - DEPTHS[0])
    average, quantity = first_rung_average()
    target = average * (1 + TAKE_PROFIT)
    out = deal(bars((100.0, rung - 0.01, rung + 0.1), (target + 0.01, rung, target)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", FREE)
    assert out["exit"] == "take-profit"
    assert out["rungs"] == 1
    assert out["equity"] == pytest.approx(1 + quantity * target - ENTRY * (1 + FIRST_RUNG))


def test_a_minute_that_reaches_a_rung_and_the_take_profit_fills_only_the_rung():
    out = deal(bars((102.0, 97.3, 100.0), (100.5, 99.5, 100.0)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", FREE)
    assert out["rungs"] == 1
    assert out["end"] == MIDNIGHT + 2 * MINUTE


def test_the_ladder_stops_where_the_position_costs_the_cap():
    out = deal(bars((100.0, 80.0, 80.5)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", FREE)
    assert out["peak"] == pytest.approx(CAP)
    assert out["rungs"] == 6
    assert out["exit"] == "delisted"


def test_unwind_sells_a_72nd_of_the_capped_position_at_each_hourly_close_under_the_average():
    path = [(MIDNIGHT, 100.0, 80.0, 86.0)] + [(MIDNIGHT + k * MINUTE, 86.5, 85.5, 86.0) for k in range(1, 120)]
    capped = deal(path[:1], MIDNIGHT - MINUTE, 100.0, 1.0, "hold", FREE, until=MIDNIGHT + MINUTE)["quantity"]
    out = deal(path, MIDNIGHT - MINUTE, 100.0, 1.0, "unwind", FREE, until=MIDNIGHT + 120 * MINUTE)
    assert out["exit"] == "open"
    assert out["quantity"] == pytest.approx(capped * (1 - 2 / UNWIND_HOURS))


def test_the_stop_sells_everything_at_the_first_close_under_its_line_once_capped():
    out = deal(bars((100.0, 80.0, 86.0), (86.0, 70.0, 75.0)), MIDNIGHT - MINUTE, 100.0, 1.0, "stop", FREE)
    assert out["exit"] == "stop"
    assert out["end"] == MIDNIGHT + 2 * MINUTE
    assert out["quantity"] == 0


def test_a_long_pays_positive_funding_on_its_position_at_each_print():
    out = deal(bars((100.5, 99.5, 100.0)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", FREE,
               prints=[(MIDNIGHT, 0.001)], until=MIDNIGHT + MINUTE)
    assert out["funding"] == pytest.approx(0.001 * ENTRY)
    assert out["equity"] == pytest.approx(1 - 0.001 * ENTRY)


def test_entries_and_forced_sales_pay_taker_while_rungs_and_take_profits_pay_maker():
    costs = Costs(taker=0.001, maker=0.0002)
    target = 100 * (1 + TAKE_PROFIT)
    out = deal(bars((target + 0.01, 99.9, 101.0)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", costs)
    assert out["fees"] == pytest.approx(ENTRY * 0.001 + ENTRY / 100 * target * 0.0002)
    gone = deal(bars((100.5, 99.5, 99.0)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", costs)
    assert gone["fees"] == pytest.approx(ENTRY * 0.001 + ENTRY / 100 * 99.0 * 0.001)


def test_equity_at_or_below_zero_is_ruin():
    out = deal(bars((100.0, 80.0, 86.0), (86.0, 1.0, 1.0)), MIDNIGHT - MINUTE, 100.0, 1.0, "hold", FREE)
    assert out["exit"] == "ruin"


def test_equity_is_marked_at_each_utc_midnight():
    path = [(MIDNIGHT - MINUTE, 100.5, 99.5, 99.0), (MIDNIGHT, 100.5, 99.5, 100.0)]
    out = deal(path, MIDNIGHT - 2 * MINUTE, 100.0, 1.0, "hold", FREE, until=MIDNIGHT + MINUTE)
    assert out["marks"] == [(MIDNIGHT, pytest.approx(1 - ENTRY * 0.01))]
