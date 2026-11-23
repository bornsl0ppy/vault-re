"""What does a vault do? Position path, episodes, entry context and add behaviour, from its fills and BTC
candles. Read-only.

    python describe.py convexity
"""
import bisect
import json
import math
import sys
from collections import Counter

import numpy as np

name = sys.argv[1]
fills = json.load(open(f"data/{name}/fills_ordered.json"))      # execution order, chained by pull.ordered()
candles = json.load(open(f"data/{name}/btc_15m_binance.json"))
t_c = np.array([c["t"] for c in candles])
close = np.array([c["c"] for c in candles])
high = np.array([c["h"] for c in candles])
low = np.array([c["l"] for c in candles])


def at(ms):
    """Index of the last closed 15m candle before ms."""
    return bisect.bisect_right(t_c, ms - 900_000) - 1


def signed(f):
    return float(f["sz"]) * (1 if f["side"] == "B" else -1)


# Position path and continuity
breaks = 0
pos = []
for prev, f in zip([None] + fills[:-1], fills):
    start = float(f["startPosition"])
    if prev is not None and abs(start - (float(prev["startPosition"]) + signed(prev))) > 1e-6:
        breaks += 1
    pos.append(start + signed(f))
pos = np.array(pos)
times = np.array([f["time"] for f in fills])
dt_h = np.diff(np.append(times, times[-1])) / 3.6e6
held = pos
total_h = dt_h.sum()
print(f"{len(fills)} fills, {total_h/24:.0f} days; position breaks between fills: {breaks}")
print(f"time long {dt_h[held > 1e-9].sum()/total_h*100:.0f}%, short {dt_h[held < -1e-9].sum()/total_h*100:.0f}%, flat {dt_h[np.abs(held) <= 1e-9].sum()/total_h*100:.0f}%")
print(f"|position| BTC: median {np.median(np.abs(pos)):.2f}, p90 {np.percentile(np.abs(pos), 90):.2f}, max {np.abs(pos).max():.2f}")
dirs = Counter(f["dir"] for f in fills)
print("fill directions:", dict(dirs))
maker = Counter((f["dir"].split()[0], not f["crossed"]) for f in fills)
print("maker share: open", round(maker[("Open", True)] / max(1, maker[("Open", True)] + maker[("Open", False)]) * 100), "% | close",
      round(maker[("Close", True)] / max(1, maker[("Close", True)] + maker[("Close", False)]) * 100), "%")

# Episodes: from flat (or a flip) to flat (or the next flip)
episodes, cur = [], None
for i, f in enumerate(fills):
    before, after = float(f["startPosition"]), pos[i]
    if cur is None and abs(after) > 1e-9:
        cur = dict(side=np.sign(after), start=f["time"], first=i, fills=[], peak=0.0)
    if cur is not None:
        cur["fills"].append(i)
        cur["peak"] = max(cur["peak"], abs(after))
        flipped = np.sign(after) != cur["side"] and abs(after) > 1e-9
        if abs(after) <= 1e-9 or flipped:
            cur["end"] = f["time"]
            episodes.append(cur)
            cur = dict(side=np.sign(after), start=f["time"], first=i, fills=[i], peak=abs(after)) if flipped else None
if cur is not None:
    cur["end"] = None
    episodes.append(cur)

rows = []
for e in episodes:
    idx = e["fills"]
    pnl = sum(float(fills[i]["closedPnl"]) - float(fills[i]["fee"]) for i in idx)
    k = at(e["start"])
    if k < 7 * 96:
        continue
    px = close[k]
    window = slice(k - 7 * 96, k + 1)
    rng = (px - low[window].min()) / max(high[window].max() - low[window].min(), 1e-9)
    adds_in_profit = adds = 0
    vwap_num = vwap_den = 0.0
    for i in idx:
        f = fills[i]
        if not f["dir"].startswith("Open") and "Long >" not in f["dir"] and "Short >" not in f["dir"]:
            continue
        p, s = float(f["px"]), float(f["sz"])
        if vwap_den > 0:
            adds += 1
            avg = vwap_num / vwap_den
            adds_in_profit += (p > avg) if e["side"] > 0 else (p < avg)
        vwap_num += p * s
        vwap_den += s
    rows.append(dict(side=e["side"], days=((e["end"] or times[-1]) - e["start"]) / 8.64e7, peak=e["peak"], fills=len(idx), pnl=pnl,
                     ret4h=px / close[k - 16] - 1, ret24h=px / close[k - 96] - 1, ret72h=px / close[k - 288] - 1, range7d=rng,
                     adds=adds, adds_in_profit=adds_in_profit, open=e["end"] is None))

print(f"\n{len(rows)} episodes ({sum(r['open'] for r in rows)} still open)")
for side, label in ((1, "long"), (-1, "short")):
    s = [r for r in rows if r["side"] == side]
    if not s:
        continue
    pnl = np.array([r["pnl"] for r in s])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    adds = sum(r["adds"] for r in s)
    print(f"{label:5s}: n {len(s):3d}, median {np.median([r['days'] for r in s]):.1f} d, peak {np.median([r['peak'] for r in s]):.2f} BTC, "
          f"win {len(wins)/len(s)*100:3.0f}%, avg win ${wins.mean() if len(wins) else 0:,.0f}, avg loss ${losses.mean() if len(losses) else 0:,.0f}, total ${pnl.sum():,.0f}")
    print(f"       entry context (median): 4h {np.median([r['ret4h'] for r in s])*1e4:+.0f} bps, 24h {np.median([r['ret24h'] for r in s])*1e4:+.0f} bps, "
          f"72h {np.median([r['ret72h'] for r in s])*1e4:+.0f} bps, place in 7-day range {np.median([r['range7d'] for r in s]):.2f} (0 low, 1 high)")
    print(f"       adds: {adds}, made while the position was winning: {sum(r['adds_in_profit'] for r in s)/max(adds,1)*100:.0f}%")
pnl_all = np.array([r["pnl"] for r in rows])
srt = np.sort(pnl_all)[::-1]
print(f"\nall episodes: total ${pnl_all.sum():,.0f}; top 5 episodes ${srt[:5].sum():,.0f} ({srt[:5].sum()/pnl_all.sum()*100:.0f}% of total); "
      f"skew {((pnl_all - pnl_all.mean())**3).mean() / pnl_all.std()**3:.2f}")
json.dump(rows, open(f"data/{name}/episodes.json", "w"), indent=1, default=float)
