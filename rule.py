"""Astraeus's +convexity rule on hourly BTC bars: target leverage = c x trend x participation / variance, reached
STEP of the way each night. Bars sit on a gapless hourly clock."""
import json

import numpy as np

HOUR, DAY = 3_600_000, 86_400_000
STEP = 0.8


def hourly(path):
    """Binance klines on a gapless hourly clock: open times, closes, quote volumes and the count of missing bars.
    A missing bar carries the close forward with zero volume."""
    rows = json.load(open(path))
    opens = np.arange(rows[0][0], rows[-1][0] + HOUR, HOUR)
    at = (np.array([r[0] for r in rows]) - opens[0]) // HOUR
    close = np.full(len(opens), np.nan)
    close[at] = [float(r[4]) for r in rows]
    volume = np.zeros(len(opens))
    volume[at] = [float(r[7]) for r in rows]
    last_seen = np.maximum.accumulate(np.where(np.isfinite(close), np.arange(len(close)), 0))
    return opens, close[last_seen], volume, len(opens) - len(rows)


def on_clock(opens, other_opens, values):
    """Values of a shorter series placed on the opens clock, NaN outside it."""
    out = np.full(len(opens), np.nan)
    out[(other_opens - opens[0]) // HOUR] = values
    return out


def ema(x, span):
    alpha, out = 2 / (span + 1), np.empty(len(x))
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def rolling_sum(x, n):
    """Sum of the last n values; NaN while the window holds a missing one."""
    total = np.concatenate([[0.0], np.cumsum(np.nan_to_num(x))])
    missing = np.concatenate([[0], np.cumsum(np.isnan(x))])
    out = np.full(len(x), np.nan)
    out[n - 1:] = np.where(missing[n:] > missing[:-n], np.nan, total[n:] - total[:-n])
    return out


def trend(close, fast=48, slow=480):
    return np.log(ema(close, fast) / ema(close, slow))


def variance(close):
    """Annualized EWMA (span 336 h) of hourly squared log returns."""
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    return 8760 * ema(returns ** 2, 336)


def participation(volume):
    """Last 24 hours' volume over the 30-day daily mean."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return rolling_sum(volume, 24) / (rolling_sum(volume, 720) / 30)


def daily_participation(volume):
    """A day's volume over the mean of the 30 days ending with it."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return volume / (rolling_sum(volume, 30) / 30)
