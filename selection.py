"""How a grid vault picks its coins: each fresh entry (a coin it hasn't traded for FRESH_HOURS) ranked against every
Hyperliquid perp with candles at that moment, by recent move, volume and volatility. Then how often simple rules
would have picked the vault's coin, at every deal start. Read-only.

    python selection.py cold_process
"""
import collections
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HOUR = 3_600_000
FRESH_HOURS = 6
BAR = 4 * HOUR
NEVER = {"BTC", "ETH", "SOL"}

name = sys.argv[1]
fills = [f for f in json.load(open(f"data/{name}/fills_ordered.json")) if f["dir"] in ("Open Long", "Close Long")]
series = {}
for path in Path("data/candles_4h").glob("*.json"):
    rows = json.load(open(path))
    if rows:
        series[path.stem] = (np.array([r["T"] + 1 for r in rows]), np.array([float(r["c"]) for r in rows]),
                             np.array([float(r["v"]) * float(r["c"]) for r in rows]))


def features(t):
    """Per coin, from its 4h bars closed by t: 4h, 24h and 7d moves, 24h $ volume, 7d volatility of 4h returns."""
    out = {}
    for coin, (close_t, close, usd) in series.items():
        i = np.searchsorted(close_t, t, side="right") - 1
        if i < 42 or close_t[i] < t - BAR:
            continue
        logs = np.diff(np.log(close[i - 42:i + 1]))
        out[coin] = {"move 4h": close[i] / close[i - 1] - 1, "move 24h": close[i] / close[i - 6] - 1,
                     "move 7d": close[i] / close[i - 42] - 1, "volume 24h": usd[i - 5:i + 1].sum(),
                     "volatility 7d": logs.std()}
    return out


def deal_starts():
    """Every entry from flat, flagged fresh when the coin wasn't traded in the FRESH_HOURS before (a pick, not a
    restart after a take-profit)."""
    last_seen, out = {}, []
    for f in fills:
        if float(f["startPosition"]) == 0 and f["side"] == "B":
            out.append((f["time"], f["coin"], f["time"] - last_seen.get(f["coin"], -10**15) > FRESH_HOURS * HOUR))
        last_seen[f["coin"]] = f["time"]
    return out


def top_by_volume(market, n):
    return sorted(market, key=lambda c: -market[c]["volume 24h"])[:n]


PROXIES = {
    "most 24h volume": lambda m: top_by_volume(m, len(m)),
    "top 10 by volume, most volatile 7d": lambda m: sorted(top_by_volume(m, 10), key=lambda c: -m[c]["volatility 7d"]),
    "top 20 by volume, most volatile 7d": lambda m: sorted(top_by_volume(m, 20), key=lambda c: -m[c]["volatility 7d"]),
    "top 20 by volume, best 24h move": lambda m: sorted(top_by_volume(m, 20), key=lambda c: -m[c]["move 24h"]),
    "top 20 by volume, best 7d move": lambda m: sorted(top_by_volume(m, 20), key=lambda c: -m[c]["move 7d"]),
}


def proxy_hits(starts):
    """Per rule, the share of deal starts where the vault's coin was the rule's first pick, or in its first three
    (BTC, ETH and SOL left out, as the vault never trades them)."""
    hits = {name: [0, 0] for name in PROXIES}
    n = 0
    for t, coin, _ in starts:
        market = {c: v for c, v in features(t).items() if c not in NEVER}
        if coin not in market:
            continue
        n += 1
        for name, rank in PROXIES.items():
            order = rank(market)
            hits[name][0] += order[:1] == [coin]
            hits[name][1] += coin in order[:3]
    return n, hits


starts = deal_starts()
entries = [(t, coin) for t, coin, fresh in starts if fresh]
ranks, tops, sizes = collections.defaultdict(list), collections.Counter(), []
for t, coin in entries:
    market = features(t)
    if coin not in market:
        continue
    sizes.append(len(market))
    for key in market[coin]:
        values = np.array([m[key] for m in market.values()])
        ranks[key].append(float((values < market[coin][key]).mean()))
    by_move = sorted(market, key=lambda c: -market[c]["move 24h"])
    tops["top 5 by 24h move"] += coin in by_move[:5]
    tops["top 10 by 24h move"] += coin in by_move[:10]
    tops["top 20 by 24h volume"] += coin in sorted(market, key=lambda c: -market[c]["volume 24h"])[:20]
n = len(ranks["move 24h"])
print(f"{len(entries)} fresh entries ({FRESH_HOURS} h without the coin), {n} ranked against a median of {statistics.median(sizes):.0f} perps")
for key, values in ranks.items():
    print(f"  {key:14} percentile among perps: median {statistics.median(values):.0%}, p25 {np.percentile(values, 25):.0%}, "
          f"p75 {np.percentile(values, 75):.0%}; in the top tenth {np.mean(np.array(values) >= 0.9):.0%}")
for key, count in tops.items():
    print(f"  {key}: {count / n:.0%}")
coins = collections.Counter(c for _, c in entries)
print("  fresh entries per coin: " + ", ".join(f"{c} {k}" for c, k in coins.most_common(12)))
hours = collections.Counter(dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).hour for t, _ in entries)
print("  by UTC hour: " + ", ".join(f"{h:02d}h {k}" for h, k in sorted(hours.items())))
for label, subset in (("fresh picks", [s for s in starts if s[2]]), ("all deal starts", starts)):
    n, hits = proxy_hits(subset)
    print(f"  rules against the vault's {label} (n {n}): " + "; ".join(
        f"{name} {first / n:.0%} first, {three / n:.0%} top 3" for name, (first, three) in hits.items()))
