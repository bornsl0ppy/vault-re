"""A multi-coin vault's fingerprint from its fills, orders and TWAPs: what it trades, how it executes, when, how
directional it is, how its positions run, and whether it enters with or against the recent move. Descriptive, to
decide whether the vault is worth decoding. Read-only.

    python fingerprint.py delta_01
"""
import bisect
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import numpy as np

from pull import is_perp, signed

HOUR, DAY = 3_600_000, 86_400_000
LOOKBACKS_H = (24, 72, 168)
PAIR_WINDOW = HOUR

name = sys.argv[1]
fills = json.load(open(f"data/{name}/fills_ordered.json"))
orders = json.load(open(f"data/{name}/orders.json"))
twaps = json.load(open(f"data/{name}/twaps.json"))
breaks = json.load(open(f"data/{name}/breaks.json"))
details = json.load(open(f"data/{name}/details.json"))
perp = [f for f in fills if is_perp(f["coin"])]


def day(ms):
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).strftime("%Y-%m-%d")


def closes(coin, cache={}):
    """A coin's 4h closes as (close times, prices), or None without candles."""
    if coin not in cache:
        path = Path(f"data/candles_4h/{coin}.json")
        rows = json.load(open(path)) if path.exists() else []
        cache[coin] = (np.array([r["T"] + 1 for r in rows]), np.array([float(r["c"]) for r in rows])) if rows else None
    return cache[coin]


def price_at(coin, t):
    series = closes(coin)
    if series is None:
        return None
    i = np.searchsorted(series[0], t, side="right") - 1
    return float(series[1][i]) if i >= 0 else None


def overview():
    traded = collections.Counter()
    for f in perp:
        traded[f["coin"]] += float(f["sz"]) * float(f["px"])
    final = {}
    for row in twaps:                             # an activation row, then a finished or terminated one; old ones lack ids
        key = (row["state"]["coin"], row["state"]["timestamp"])
        if key not in final or row["status"]["status"] != "activated":
            final[key] = row
    twap_ntl = sum(float(r["state"]["executedNtl"]) for r in final.values())
    total = sum(traded.values()) + twap_ntl
    maker = sum(float(f["sz"]) * float(f["px"]) for f in perp if not f["crossed"]) / max(sum(traded.values()), 1)
    print(f"== {details['name']} ({name}): TVL ${float(details['portfolio'][0][1]['accountValueHistory'][-1][1]):,.0f}, "
          f"leader holds {details['leaderFraction']:.0%}, deposits {'open' if details.get('allowDeposits') else 'closed'}")
    print(f"{len(fills):,} fills ({len(fills) - len(perp)} spot) from {day(fills[0]['time'])} to {day(fills[-1]['time'])}; "
          f"{len(traded)} perp coins; ${total:,.0f} traded, {twap_ntl / total:.0%} of it through {len(final)} TWAPs; "
          f"maker on {maker:.0%} of the non-TWAP notional")
    print("top coins by notional: " + ", ".join(f"{c} {v / total:.0%}" for c, v in traded.most_common(10)))
    minutes = [r["state"]["minutes"] for r in final.values()]
    if minutes:
        print(f"TWAPs: median {statistics.median(minutes):.0f} min (p90 {np.percentile(minutes, 90):.0f}), "
              f"randomized {sum(r['state']['randomize'] for r in final.values()) / len(final):.0%}, "
              f"reduce-only {sum(r['state']['reduceOnly'] for r in final.values()) / len(final):.0%}")
    print(f"perp position breaks: {len(breaks)} (fills the API leaves out; TWAP slices among them)")


def execution():
    latest = {}
    for row in orders:
        latest[row["order"]["oid"]] = row["order"]
    tif = collections.Counter(o.get("tif") or o["orderType"] for o in latest.values())
    cloids = [o["cloid"] for o in latest.values() if o.get("cloid")]
    heads = collections.Counter(c[:10] for c in cloids)
    placed = [o["timestamp"] for o in latest.values()]
    print(f"\nlatest {len(latest)} orders ({day(min(placed))} to {day(max(placed))}): " +
          ", ".join(f"{k} {v / len(latest):.0%}" for k, v in tif.most_common()) +
          f"; reduce-only {sum(o['reduceOnly'] for o in latest.values()) / len(latest):.0%}, "
          f"triggers {sum(o['isTrigger'] for o in latest.values()) / len(latest):.0%}, with a cloid {len(cloids) / len(latest):.0%}")
    if heads:
        print("  cloid heads: " + ", ".join(f"{h} x{n}" for h, n in heads.most_common(6)))
    hours = collections.Counter(dt.datetime.fromtimestamp(f["time"] / 1000, dt.timezone.utc).hour for f in perp)
    top = hours.most_common(3)
    print("  fills by UTC hour, busiest: " + ", ".join(f"{h:02d}h {n / len(perp):.0%}" for h, n in top) +
          f" (flat would be 4% an hour)")


def positions_at(t, chains):
    """Each coin's position at time t from its fill chain: the last known position before t."""
    out = {}
    for coin, (times, after) in chains.items():
        i = bisect.bisect_left(times, t) - 1
        if i >= 0 and after[i] != 0:
            out[coin] = after[i]
    return out


def exposure():
    chains = collections.defaultdict(lambda: ([], []))
    for f in perp:
        times, after = chains[f["coin"]]
        times.append(f["time"])
        after.append(float(f["startPosition"]) + signed(f))
    history = dict(details["portfolio"])["allTime"]["accountValueHistory"]
    av_t, av = [t for t, _ in history], [float(v) for _, v in history]
    rows = []
    for t in range((fills[0]["time"] // DAY + 1) * DAY, fills[-1]["time"], DAY):
        held = positions_at(t, chains)
        notional = {c: q * p for c, q in held.items() if (p := price_at(c, t)) is not None}
        i = bisect.bisect_right(av_t, t) - 1
        if not notional or i < 0 or av[i] <= 0:
            continue
        gross, net = sum(abs(v) for v in notional.values()), sum(notional.values())
        rows.append((t, len(notional), gross / av[i], net / gross, sum(v for v in notional.values() if v > 0) / gross))
    if not rows:
        return
    _, held, lev, net_share, long_share = zip(*rows)
    print(f"\ndaily snapshots ({len(rows)} days with positions): coins held median {statistics.median(held):.0f} "
          f"(p90 {np.percentile(held, 90):.0f}); gross leverage median {statistics.median(lev):.2f}x (p90 {np.percentile(lev, 90):.2f}x)")
    print(f"  net / gross: median {statistics.median(net_share):+.2f}; |net| under 20% of gross on "
          f"{np.mean(np.abs(net_share) < 0.2):.0%} of days, all one side on {np.mean(np.abs(net_share) > 0.95):.0%}; "
          f"longs are {statistics.median(long_share):.0%} of gross (median)")


def episodes():
    """Per coin, from flat to flat (or a flip): known from each fill's startPosition, so a TWAP the API leaves out
    still shows where the position went."""
    out = []
    for coin in sorted({f["coin"] for f in perp}):
        open_ = None
        for f in (x for x in perp if x["coin"] == coin):
            before, after = float(f["startPosition"]), float(f["startPosition"]) + signed(f)
            if open_ and before == 0:
                open_["end"] = f["time"]
                out.append(open_)
                open_ = None
            if open_ is None and after != 0:
                open_ = {"coin": coin, "side": 1 if after > 0 else -1, "start": f["time"], "entry": float(f["px"]),
                         "pnl": 0.0, "adds": 0, "adds_winning": 0}
                continue
            if open_ is None:
                continue
            open_["pnl"] += float(f["closedPnl"])
            if abs(after) > abs(before) and after * before > 0:
                open_["adds"] += 1
                open_["adds_winning"] += (float(f["px"]) - open_["entry"]) * open_["side"] > 0
            if after == 0 or after * open_["side"] < 0:
                open_["end"] = f["time"]
                out.append(open_)
                open_ = {"coin": coin, "side": -open_["side"], "start": f["time"], "entry": float(f["px"]),
                         "pnl": 0.0, "adds": 0, "adds_winning": 0} if after != 0 else None
    return out


def describe(eps):
    hours = [(e["end"] - e["start"]) / HOUR for e in eps]
    print(f"\n{len(eps)} closed episodes: holding median {statistics.median(hours):.1f} h "
          f"(p25 {np.percentile(hours, 25):.1f}, p75 {np.percentile(hours, 75):.1f}); long {np.mean([e['side'] > 0 for e in eps]):.0%}")
    for lookback in LOOKBACKS_H:
        moves = []
        for e in eps:
            now, then = price_at(e["coin"], e["start"]), price_at(e["coin"], e["start"] - lookback * HOUR)
            if now and then:
                moves.append((now / then - 1) * e["side"] * 1e4)
        if moves:
            print(f"  entry vs the coin's prior {lookback:>3} h, signed by side: median {statistics.median(moves):+.0f} bps, "
                  f"with the move {np.mean(np.array(moves) > 0):.0%} (n {len(moves)})")
    adds = sum(e["adds"] for e in eps)
    if adds:
        print(f"  adds: {adds}, made while winning {sum(e['adds_winning'] for e in eps) / adds:.0%}")
    starts = sorted((e["start"], e["side"], e["coin"]) for e in eps)
    paired = sum(1 for t, side, coin in starts
                 if any(abs(t2 - t) <= PAIR_WINDOW and s2 == -side and c2 != coin for t2, s2, c2 in starts))
    print(f"  entries with an opposite-side entry in another coin within an hour: {paired / len(starts):.0%}")
    pnl = sorted((e["pnl"] for e in eps), reverse=True)
    total = sum(pnl)
    print(f"  realized P&L ${total:,.0f} (fills only; TWAP exits missing), winners {np.mean(np.array(pnl) > 0):.0%}, "
          f"top 5 episodes {sum(pnl[:5]) / total:.0%} of it" if total else "")


overview()
execution()
exposure()
describe(episodes())
