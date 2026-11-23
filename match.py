"""Which textbook trend signal does the vault's position follow? Time-weighted agreement between the sign of
its BTC position and the sign of each candidate signal, on hourly closes. Read-only.

    python match.py convexity
"""
import bisect
import json
import sys

import numpy as np

name = sys.argv[1]
fills = json.load(open(f"data/{name}/fills_ordered.json"))
candles = json.load(open(f"data/{name}/btc_15m_binance.json"))
t15 = np.array([c["t"] for c in candles])
c15 = np.array([c["c"] for c in candles])
hourly = t15 % 3_600_000 == 2_700_000                  # the last 15m candle of each hour
t = t15[hourly] + 900_000                            # hour close time
close = c15[hourly]

ftimes = [f["time"] for f in fills]
pos_after = [float(f["startPosition"]) + float(f["sz"]) * (1 if f["side"] == "B" else -1) for f in fills]
i = np.array([bisect.bisect_right(ftimes, x) - 1 for x in t])
live = i >= 0
position = np.where(live, np.array(pos_after)[np.clip(i, 0, None)], 0.0)
held = live & (np.abs(position) > 1e-9)
side = np.sign(position)


def ema(x, n):
    out, a = np.empty_like(x), 2 / (n + 1)
    out[0] = x[0]
    for k in range(1, len(x)):
        out[k] = a * x[k] + (1 - a) * out[k - 1]
    return out


signals = {}
for n in (24, 48, 72, 120, 168, 240, 336, 504):
    signals[f"close vs EMA{n}h"] = np.sign(close - ema(close, n))
for f_, s_ in ((12, 48), (24, 96), (24, 168), (48, 168), (72, 240), (120, 336), (96, 336), (144, 336), (120, 504),
               (168, 504), (168, 672), (240, 720), (336, 1008)):
    signals[f"EMA{f_}h vs EMA{s_}h"] = np.sign(ema(close, f_) - ema(close, s_))
for n in (24, 48, 72, 120, 168, 336):
    r = np.full_like(close, np.nan)
    r[n:] = close[n:] / close[:-n] - 1
    signals[f"{n}h return"] = np.sign(r)

warm = np.arange(len(close)) >= 1008
half = np.median(np.nonzero(held & warm)[0])
first, second = np.arange(len(close)) < half, np.arange(len(close)) >= half
rows = []
for label, sig in signals.items():
    ok = held & warm & ~np.isnan(sig)
    a1, a2 = (np.mean(side[ok & part] == sig[ok & part]) for part in (first, second))
    rows.append((a1, a2, label))
rows.sort(reverse=True)
best = rows[0]
print(f"picked on the first half: {best[2]} ({best[0]*100:.1f}% there); on the second half it agrees {best[1]*100:.1f}%")
print(f"hours with a position: {(held & warm).sum()} of {warm.sum()}; short share {np.mean(side[held & warm] < 0)*100:.0f}%")
print("always short would agree", f"{np.mean(side[held & warm] < 0)*100:.0f}%")
for a1, a2, label in rows[:12]:
    print(f"  {label:20s} first half {a1*100:5.1f}%  second half {a2*100:5.1f}%")
