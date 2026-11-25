"""The two charts in the README, from what the other scripts save. Read-only.

    python charts.py      # -> charts/*.png, after backtest.py's data and grid_backtest.py holdout
"""
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, LogLocator

import backtest
import grid_backtest as grid
from rule import HOUR

SURFACE, INK, SECONDARY, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"
OUT = Path("charts")


def figure(title, subtitle):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=8, length=0, pad=6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}x" if v >= 1 else f"{v:g}x"))
    fig.text(0.05, 0.965, title, color=INK, fontsize=11.5, fontweight="bold", va="top")
    fig.text(0.05, 0.905, subtitle, color=SECONDARY, fontsize=8.5, va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.09, right=0.84, top=0.74, bottom=0.08)
    return fig, ax


def line(ax, times, values, color, label, end_label, width=1.6, rise=0):
    """A series with an end dot and its end label, `rise` points above the dot (negative: below)."""
    when = np.array(times, dtype="datetime64[ms]")
    ax.plot(when, values, color=color, linewidth=width, solid_capstyle="round", solid_joinstyle="round", label=label)
    ax.plot(when[-1], values[-1], "o", color=color, markersize=5.5, markeredgecolor=SURFACE, markeredgewidth=1.3)
    ax.annotate(end_label, (when[-1], values[-1]), xytext=(7, rise), textcoords="offset points", va="center",
                fontsize=8, color=SECONDARY)


def finish(fig, ax, name):
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "BTC held")
    ax.legend([handles[i] for i in order], [labels[i] for i in order], loc="lower left", bbox_to_anchor=(0, 1.02),
              ncol=2, frameon=False, fontsize=8, handlelength=1.6, labelcolor=SECONDARY, borderaxespad=0)
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.png", facecolor=SURFACE)
    plt.close(fig)


def drawdown(equity):
    return (equity / np.maximum.accumulate(equity) - 1).min()


def convexity():
    """+convexity's rule as one continuous book on Binance inputs, the vault's own months shaded."""
    market = backtest.Market()
    mask = market.between(backtest.REGIMES[0][0], backtest.REGIMES[-1][1])
    returns, _ = backtest.run(market, backtest.scales(market), mask, "spot", "binance", backtest.READS["binance"]["fee"])
    fills = market.days[mask] + 2 * HOUR
    times = np.concatenate([fills[:1], fills + 24 * HOUR])
    rule = np.concatenate([[1.0], np.cumprod(1 + returns)])
    btc = np.concatenate([[1.0], np.cumprod(1 + market.day_return[mask])])
    fig, ax = figure(f"+convexity's rule since 2017: a worst drawdown of {drawdown(rule):.0%}, against {drawdown(btc):.0%} "
                     "for BTC",
                     "Binance data, fees and funding in. Shaded: the vault's own months, the best run the rule's shorts "
                     "ever had.")
    seen = [np.datetime64(backtest.ms(day), "ms") for day in backtest.SEEN]
    ax.axvspan(*seen, color=GRID, alpha=0.7, linewidth=0, zorder=0)
    line(ax, times, btc, MUTED, "BTC held", f"BTC {btc[-1] - 1:+,.0%}", width=1.2, rise=-8)
    line(ax, times, rule, BLUE, "The rule", f"rule {rule[-1] - 1:+,.0%}", rise=8)
    finish(fig, ax, "convexity")


def cold_process():
    """Cold Process's rule from 2020 as decoded, and with a stop."""
    runs, _ = pickle.load(open("data/cold_process/grid_holdout_deals.pkl", "rb"))
    days, equity, _ = grid.daily(runs["centre"], *grid.HOLDOUT)
    stopped = grid.summary(runs["stop"], *grid.HOLDOUT)
    fig, ax = figure(f"Cold Process's rule from 2020: up {equity.max():,.0f}x, then LUNA took it to zero",
                     "Binance minute data with delisted coins, fees and funding in, compounding with no limit on size.\n"
                     f"With a stop 10% under its average it survives, after falling {-stopped['drawdown']:.0%}.")
    for color, name, label in ((BLUE, "centre", "As decoded"), (ORANGE, "stop", "With the stop")):
        days, equity, _ = grid.daily(runs[name], *grid.HOLDOUT)
        alive = np.flatnonzero(equity > 0)
        last = runs[name][-1]
        if last["exit"] == "ruin":
            end_label = f"zero on {np.datetime64(last['end'], 'ms').astype('datetime64[D]')}"
        else:
            end_label = f"{grid.summary(runs[name], *grid.HOLDOUT)['total'] * 100:+,.0f}%"
        line(ax, days[alive], equity[alive], color, label, end_label)
    finish(fig, ax, "cold-process")


if __name__ == "__main__":
    for old in OUT.glob("*.png"):
        old.unlink()
    convexity()
    cold_process()
    print(f"{len(list(OUT.glob('*.png')))} charts in {OUT}/")
