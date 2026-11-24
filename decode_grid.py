"""Decode a long-only averaging-down grid vault (Cold Process): its ladders from the latest orders, then over the
whole fill history each trade's entry, adds (size against the one before, depth below the entry), take-profit over
the average entry, and how deep and large positions got. Read-only.

    python decode_grid.py cold_process
"""
import bisect
import collections
import datetime as dt
import json
import statistics
import sys

import numpy as np

from pull import signed

HOUR = 3_600_000
LADDER_WINDOW = 10 * 60_000                   # buy orders placed this soon after an entry belong to its ladder

name = sys.argv[1]
fills = [f for f in json.load(open(f"data/{name}/fills_ordered.json")) if f["dir"] in ("Open Long", "Close Long")]
orders = json.load(open(f"data/{name}/orders.json"))
details = json.load(open(f"data/{name}/details.json"))
history = dict(details["portfolio"])["allTime"]["accountValueHistory"]
av_t, av = [t for t, _ in history], [float(v) for _, v in history]


def account_value(t):
    """The vault's account value at t; its first positive one before the history starts counting."""
    i = bisect.bisect_right(av_t, t) - 1
    return next(v for v in av[max(i, 0):] if v > 0)


def q(values, p):
    return float(np.percentile(values, p)) if values else float("nan")


def ladders():
    """The latest orders: after each entry, the resting buys below it and the take-profit sell above."""
    latest = {}
    for row in orders:
        latest[row["order"]["oid"]] = row["order"]
    by_coin = collections.defaultdict(list)
    for o in latest.values():
        by_coin[o["coin"]].append(o)
    depths, ratios, tps = collections.defaultdict(list), [], []
    for coin, group in by_coin.items():
        group.sort(key=lambda o: o["timestamp"])
        for i, o in enumerate(group):
            if o["side"] != "A" or not o["reduceOnly"]:
                continue
            ladder = [b for b in group if b["side"] == "B" and 0 <= b["timestamp"] - o["timestamp"] <= LADDER_WINDOW]
            ladder.sort(key=lambda b: -float(b["limitPx"]))
            if len(ladder) < 3:
                continue
            top = float(ladder[0]["limitPx"])
            for k, b in enumerate(ladder):
                depths[k].append((1 - float(b["limitPx"]) / top) * 1e4)
            sizes = [float(b["origSz"]) for b in ladder]
            ratios += [b / a for a, b in zip(sizes, sizes[1:]) if a > 0]
    print(f"== {details['name']}: ladders in the latest {len(latest)} orders")
    print(f"  size against the buy above it: median {statistics.median(ratios):.2f}x (p10 {q(ratios, 10):.2f}, p90 {q(ratios, 90):.2f}), n {len(ratios)}")
    print("  depth below the ladder's top buy, by rung: " +
          ", ".join(f"#{k + 1} {statistics.median(v):.0f} bps" for k, v in sorted(depths.items()) if len(v) >= 5))


def trades():
    """Over all fills: from flat to flat per coin, with buys grouped by order."""
    out, open_ = [], {}
    for f in fills:
        coin = f["coin"]
        before, after = float(f["startPosition"]), float(f["startPosition"]) + signed(f)
        trade = open_.get(coin)
        if trade is None:
            if before != 0 or after <= 0:
                continue
            trade = open_[coin] = {"coin": coin, "start": f["time"], "buys": collections.OrderedDict(), "sells": [],
                                   "max_qty": 0.0, "max_ntl": 0.0, "low": float(f["px"])}
        if f["side"] == "B":
            px, sz = float(f["px"]), float(f["sz"])
            buy = trade["buys"].setdefault(f["oid"], [0.0, 0.0])
            buy[0] += px * sz
            buy[1] += sz
            trade["low"] = min(trade["low"], px)
        else:
            trade["sells"].append((float(f["px"]), float(f["sz"]), float(f["closedPnl"])))
        trade["max_qty"] = max(trade["max_qty"], after)
        trade["max_ntl"] = max(trade["max_ntl"], after * float(f["px"]))
        if after == 0:
            trade["end"] = f["time"]
            out.append(trade)
            del open_[coin]
    return out, list(open_.values())


def describe(done, still_open):
    adds_n = [len(t["buys"]) - 1 for t in done]
    depth, ratio, tp, hours, size_av, pnl = collections.defaultdict(list), [], [], [], [], []
    for t in done:
        buys = [(ntl / sz, sz) for ntl, sz in t["buys"].values()]
        base_px = buys[0][0]
        for k, (px, sz) in enumerate(buys[1:], start=1):
            depth[k].append((1 - px / base_px) * 1e4)
        ratio += [b[1] / a[1] for a, b in zip(buys[1:], buys[2:])]
        cost = sum(px * sz for px, sz in buys)
        qty = sum(sz for _, sz in buys)
        sold = sum(px * sz for px, sz, _ in t["sells"]) / sum(sz for _, sz, _ in t["sells"])
        tp.append((sold / (cost / qty) - 1) * 1e4)
        hours.append((t["end"] - t["start"]) / HOUR)
        size_av.append(t["max_ntl"] / account_value(t["start"]))
        pnl.append(sum(p for _, _, p in t["sells"]))
    print(f"\n== {len(done)} closed trades from {dt.date.fromtimestamp(done[0]['start'] / 1000)} "
          f"({len(still_open)} open now: {', '.join(t['coin'] for t in still_open) or 'none'})")
    counts = collections.Counter(adds_n)
    print("  ladder rungs filled per trade: " + ", ".join(f"{k}: {counts[k] / len(done):.0%}" for k in sorted(counts)))
    print("  depth of each filled rung below the entry: " +
          ", ".join(f"#{k} {statistics.median(v):.0f} bps (n {len(v)})" for k, v in sorted(depth.items())))
    if ratio:
        print(f"  size of each rung against the one before (from rung 2): median {statistics.median(ratio):.2f}x")
    print(f"  exit over the average entry: median {statistics.median(tp):+.0f} bps (p10 {q(tp, 10):+.0f}, p90 {q(tp, 90):+.0f}); "
          f"losing exits {np.mean(np.array(tp) < 0):.1%}")
    print(f"  holding: median {statistics.median(hours):.1f} h, p90 {q(hours, 90):.1f} h, max {max(hours):.0f} h")
    print(f"  largest position against account value: median {statistics.median(size_av):.2f}x, p90 {q(size_av, 90):.2f}x, "
          f"max {max(size_av):.2f}x")
    deep = sorted(done, key=lambda t: -t["max_ntl"] / account_value(t["start"]))[:5]
    print("  deepest trades: " + "; ".join(
        f"{t['coin']} {dt.date.fromtimestamp(t['start'] / 1000)} {len(t['buys']) - 1} rungs, "
        f"{t['max_ntl'] / account_value(t['start']):.2f}x, {(t['end'] - t['start']) / HOUR:.0f} h, ${sum(p for _, _, p in t['sells']):,.0f}"
        for t in deep))
    by_month = collections.defaultdict(list)
    for t, n, r in zip(done, adds_n, tp):
        by_month[dt.date.fromtimestamp(t["start"] / 1000).strftime("%Y-%m")].append((n, r))
    print("  by month (trades, mean rungs, median exit bps): " + "; ".join(
        f"{m} {len(v)}, {statistics.fmean(n for n, _ in v):.1f}, {statistics.median(r for _, r in v):+.0f}" for m, v in sorted(by_month.items())))
    print(f"  realized ${sum(pnl):,.0f}; worst trade ${min(pnl):,.0f}, best ${max(pnl):,.0f}")


ladders()
describe(*trades())
