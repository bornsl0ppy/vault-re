"""The vault rebalances once a day at ~01:45 UTC. Fit its nightly target exposure to daily-candle trend
signals: pick the EMA pair and the sizing shape on the first half, score them on the second. Read-only.

    python daily.py convexity
"""
import bisect
import json
import sys

import numpy as np

name = sys.argv[1]
DECIDE_MS = (1 * 60 + 45) * 60000
fills = json.load(open(f"data/{name}/fills_ordered.json"))
candles = json.load(open(f"data/{name}/btc_15m_binance.json"))
e = np.load(f"data/{name}/equity.npz")
t15 = np.array([c["t"] for c in candles])
c15 = np.array([c["c"] for c in candles])
daily_close_t = t15[(t15 % 86400000) == 86400000 - 900000] + 900000      # 00:00 UTC closes
daily_close = c15[(t15 % 86400000) == 86400000 - 900000]

ftimes = [f["time"] for f in fills]
pos_after = [float(f["startPosition"]) + float(f["sz"]) * (1 if f["side"] == "B" else -1) for f in fills]
first_day = (ftimes[0] // 86400000 + 1) * 86400000
days = np.arange(first_day, ftimes[-1], 86400000)
rows = []
for d in days:
    decided = d + DECIDE_MS + 30 * 60000                  # position once the 01:45 order has filled
    j = bisect.bisect_right(ftimes, decided) - 1
    k = bisect.bisect_right(daily_close_t, d) - 1        # the daily candle that closed at 00:00
    hi = bisect.bisect_right(e["h"], decided) - 1
    if j < 0 or k < 60 or hi < 0:
        continue
    price = e["mark"][hi]
    rows.append(dict(day=d, k=k, exposure=pos_after[j] * price / e["av"][hi]))
k = np.array([r["k"] for r in rows])
y = np.array([r["exposure"] for r in rows])
day = np.array([r["day"] for r in rows])
half = day >= np.median(day[day >= np.datetime64("2025-09-01").astype("datetime64[ms]").astype(np.int64)])
early = (day >= np.datetime64("2025-09-01").astype("datetime64[ms]").astype(np.int64)) & ~half


def ema(x, n):
    out, a = np.empty_like(x), 2 / (n + 1)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


logret = np.diff(np.log(daily_close), prepend=np.log(daily_close[0]))
vol20 = np.array([logret[max(1, i - 19):i + 1].std() for i in range(len(logret))])
results = []
for fast in (3, 4, 5, 6, 7, 10):
    for slow in (14, 18, 21, 26, 30, 42):
        if slow <= fast:
            continue
        g = ema(daily_close, fast) / ema(daily_close, slow) - 1
        z = (g / np.where(vol20 > 0, vol20, np.nan))[k]
        for part, label in ((early, "first"), (half, "second")):
            pass
        agree1 = np.mean(np.sign(z[early]) == np.sign(y[early])[...]) if early.any() else np.nan
        agree2 = np.mean(np.sign(z[half]) == np.sign(y[half]))
        c1 = np.corrcoef(z[early], y[early])[0, 1]
        c2 = np.corrcoef(z[half], y[half])[0, 1]
        results.append((c1, c2, agree1, agree2, fast, slow))
results.sort(reverse=True)
print(f"{len(y)} nightly decisions ({early.sum()} first half from Sep 2025, {half.sum()} second half)")
print("daily EMA pairs by correlation of the vol-scaled gap with the nightly exposure (first half picks):")
for c1, c2, a1, a2, f_, s_ in results[:8]:
    print(f"  EMA{f_}d vs EMA{s_}d: corr {c1:.2f} / {c2:.2f}   sign agreement {a1*100:.0f}% / {a2*100:.0f}%")

c1, c2, a1, a2, fast, slow = results[0]
g = ema(daily_close, fast) / ema(daily_close, slow) - 1
z = (g / vol20)[k]
print(f"\nshape for EMA{fast}d vs EMA{slow}d: median nightly exposure by vol-scaled gap (all nights from Sep 2025):")
m = early | half
q = np.quantile(z[m], np.linspace(0, 1, 9))
for lo, hi in zip(q, q[1:]):
    s = m & (z >= lo) & (z <= hi)
    print(f"  gap/vol {lo:+6.2f} to {hi:+6.2f}: exposure {np.median(y[s]):+.2f}x  (n {s.sum()})")
best = None
for kk in np.linspace(0.05, 1.0, 20):
    for cap in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
        pred = np.clip(kk * z, -cap, cap)
        err = np.mean((pred[early] - y[early]) ** 2)
        if best is None or err < best[0]:
            best = (err, kk, cap)
_, kk, cap = best
pred = np.clip(kk * z, -cap, cap)
r2 = lambda s: 1 - np.mean((pred[s] - y[s]) ** 2) / np.var(y[s])
print(f"fitted on the first half: exposure = clip({kk:.2f} x gap/vol, +/-{cap}); R2 first {r2(early):.2f}, second {r2(half):.2f}")
json.dump(dict(fast=fast, slow=slow, k=kk, cap=cap), open(f"data/{name}/daily_fit.json", "w"))
