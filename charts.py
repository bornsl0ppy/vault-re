"""The charts in FINDINGS.md, light and dark, from what the other scripts save. Read-only.

    python charts.py      # -> charts/*.png, after grid_backtest.py check, seen and holdout
"""
import json
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

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", series=("#2a78d6", "#eb6834", "#1baf7a")),
    "dark": dict(surface="#1a1a19", ink="#f0efec", secondary="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", series=("#3987e5", "#d95926", "#199e70")),
}
OUT = Path("charts")


def figure(theme, title, subtitle):
    fig, ax = plt.subplots(figsize=(8, 4.8), dpi=200)
    fig.patch.set_facecolor(theme["surface"])
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["axis"])
    ax.grid(axis="y", color=theme["grid"], linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(colors=theme["muted"], labelsize=8, length=0, pad=6)
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator, show_offset=False))
    fig.text(0.05, 0.965, title, color=theme["ink"], fontsize=11.5, fontweight="bold", va="top")
    fig.text(0.05, 0.91, subtitle, color=theme["secondary"], fontsize=8.5, va="top", linespacing=1.45)
    fig.subplots_adjust(left=0.09, right=0.8, top=0.72, bottom=0.08)
    return fig, ax


def line(ax, theme, times, values, color, label, end_label, width=1.6):
    when = np.array(times, dtype="datetime64[ms]")
    ax.plot(when, values, color=color, linewidth=width, solid_capstyle="round", solid_joinstyle="round", label=label)
    ax.plot(when[-1], values[-1], "o", color=color, markersize=5.5, markeredgecolor=theme["surface"], markeredgewidth=1.3)
    ax.annotate(end_label, (when[-1], values[-1]), xytext=(7, 0), textcoords="offset points", va="center",
                fontsize=8, color=theme["secondary"])


def legend(ax, theme):
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=3, frameon=False, fontsize=8, handlelength=1.6,
              labelcolor=theme["secondary"], borderaxespad=0)


def change(value, _):
    return "0%" if abs(value - 1) < 1e-9 else f"{(value - 1) * 100:+,.0f}%"


def save(fig, theme_name, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}-{theme_name}.png", facecolor=fig.get_facecolor())
    plt.close(fig)


def grid_crash(theme_name):
    """Cold Process's grid over 2020-2026: as decoded, held at the cap, and with a stop."""
    theme = THEMES[theme_name]
    runs, _ = pickle.load(open("data/cold_process/grid_holdout_deals.pkl", "rb"))
    fig, ax = figure(theme, "Cold Process's grid as a strategy, 2020 to 2026",
                     "Binance minute data with delisted coins, fees and funding. Equity compounds with no capacity "
                     "limit,\nso the peaks are fiction. The drawdowns aren't.")
    shown = (("centre", "As decoded"), ("hold", "Holding at the cap"), ("stop", "A stop 10% under the average"))
    for color, (name, label) in zip(theme["series"], shown):
        days, equity, _ = grid.daily(runs[name], *grid.HOLDOUT)
        alive = np.flatnonzero(equity > 0)
        last = runs[name][-1]
        if last["exit"] == "ruin":
            ruined = np.datetime64(last["end"], "ms").astype("datetime64[D]")
            end_label = f"ruined {ruined}, {last['coin'].removesuffix('USDT')}"
        else:
            s = grid.summary(runs[name], *grid.HOLDOUT)
            end_label = f"{s['total'] * 100:+,.0f}%\nworst drawdown {s['drawdown']:.0%}"
        line(ax, theme, days[alive], equity[alive], color, label, end_label)
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=()))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}x" if v >= 1 else f"{v:g}x"))
    legend(ax, theme)
    save(fig, theme_name, "grid-crash")


def vault_record(name, start, end):
    """A vault's flow-neutral equity from its own P&L and account value history, 1 at its first point after start."""
    history = dict(json.load(open(f"data/{name}/details.json"))["portfolio"])["allTime"]
    times = np.array([t for t, _ in history["accountValueHistory"]])
    value = np.array([float(v) for _, v in history["accountValueHistory"]])
    pnl = np.array([float(v) for _, v in history["pnlHistory"]])
    keep = np.flatnonzero((value > 0) & (times >= start) & (times <= end))
    returns = np.diff(pnl[keep]) / value[keep][:-1]
    return times[keep], np.concatenate([[1.0], np.cumprod(1 + returns)])


def grid_picks(theme_name):
    """Cold Process's months: the vault, the grid on the vault's own entries, the grid with a rule picking coins."""
    theme = THEMES[theme_name]
    fig, ax = figure(theme, "Cold Process, February to October 2026",
                     "The decoded grid makes money on the vault's own entries and loses when a rule picks the coins.\n"
                     "Grid replayed on Binance minute data with fees and funding; the vault's record is flow-neutral.")
    days, equity, _ = grid.daily(pickle.load(open("data/cold_process/grid_seen_deals.pkl", "rb")), *grid.SEEN)
    line(ax, theme, days, equity, theme["series"][0], "The grid on a rule's picks", f"{equity[-1] - 1:+.0%}")
    times, record = vault_record("cold_process", *grid.SEEN)
    line(ax, theme, times, record, theme["series"][1], "The vault", f"{record[-1] - 1:+.0%}")
    replayed = sorted(pickle.load(open("data/cold_process/grid_check_deals.pkl", "rb")), key=lambda d: d["end"])
    ends = [replayed[0]["start"]] + [d["end"] for d in replayed]
    grown = np.concatenate([[1.0], np.cumprod([d["equity"] for d in replayed])])
    line(ax, theme, ends, grown, theme["series"][2], "The grid on the vault's entries", f"{grown[-1] - 1:+.0%}")
    ax.axhline(1, color=theme["axis"], linewidth=0.8, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(change))
    legend(ax, theme)
    save(fig, theme_name, "grid-picks")


def convexity(theme_name):
    """+convexity's own months: the vault and the decoded rule, with BTC for reference."""
    theme = THEMES[theme_name]
    market = backtest.Market()
    mask = market.between(*backtest.SEEN)
    spec = backtest.READS["hl"]
    returns, _ = backtest.run(market, backtest.scales(market), mask, spec["volume"], spec["funding"], spec["fee"])
    fills = market.days[mask] + 2 * HOUR
    times = np.concatenate([fills[:1], fills + 24 * HOUR])
    fig, ax = figure(theme, "+convexity, August 2025 to November 2026",
                     "The vault's flow-neutral record and the decoded rule, run with Hyperliquid volume, funding and "
                     "4.5 bps a side.\nThe rule's scale was fitted on the first six months. BTC held for reference.")
    saved = np.load("data/convexity/equity.npz")
    record = np.concatenate([[1.0], np.cumprod(1 + np.diff(saved["pnl"]) / saved["av"][:-1])])
    record = record[np.searchsorted(saved["h"], times, side="right") - 1]
    line(ax, theme, times, record / record[0], theme["series"][0], "The vault", f"{record[-1] / record[0] - 1:+.0%}")
    rule_equity = np.concatenate([[1.0], np.cumprod(1 + returns)])
    line(ax, theme, times, rule_equity, theme["series"][1], "The decoded rule", f"{rule_equity[-1] - 1:+.0%}")
    btc = np.concatenate([[1.0], np.cumprod(1 + market.day_return[mask])])
    line(ax, theme, times, btc, theme["muted"], "BTC", f"{btc[-1] - 1:+.0%}", width=1.2)
    ax.axhline(1, color=theme["axis"], linewidth=0.8, zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(change))
    legend(ax, theme)
    save(fig, theme_name, "convexity")


if __name__ == "__main__":
    for mode in THEMES:
        grid_crash(mode)
        grid_picks(mode)
        convexity(mode)
    print(f"{len(list(OUT.glob('*.png')))} charts in {OUT}/")
