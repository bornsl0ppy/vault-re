"""How does the vault size and exit? Hourly exposure against trend strength, run since the cross, open
profit, age and volatility; order-level adds and trims; exits against the EMA cross. Read-only.

    python sizing.py convexity
"""
import bisect
import json
import sys
from collections import defaultdict

import numpy as np

name = sys.argv[1]
FAST, SLOW = 120, 504                                  # the crossover picked out of sample (match.py)
e = np.load(f"data/{name}/equity.npz")
h, pos, mark, av = e["h"], e["pos"], e["mark"], e["av"]
start = np.datetime64("2025-09-01").astype("datetime64[ms]").astype(np.int64)


def ema(x, n):
    out, a = np.empty_like(x), 2 / (n + 1)
    out[0] = x[0]
    for k in range(1, len(x)):
        out[k] = a * x[k] + (1 - a) * out[k - 1]
    return out


gap = ema(mark, FAST) / ema(mark, SLOW) - 1
r1 = np.diff(np.log(mark), prepend=np.log(mark[0]))
sigma_d = np.array([r1[max(0, i - 168):i + 1].std() * np.sqrt(24) for i in range(len(r1))])   # daily vol, trailing 7 days
signal = np.sign(gap)
cross = np.zeros(len(h), dtype=int)                     # index of the latest cross at or before each hour
last = 0
for i in range(1, len(h)):
    if signal[i] != signal[i - 1]:
        last = i
    cross[i] = last
age_d = (np.arange(len(h)) - cross) / 24
run = (mark / mark[cross] - 1) * signal                 # move since the cross, in the signal's favour
exposure = pos * mark / av
side = np.sign(pos)

ok = (h >= start) & (np.abs(pos) > 1e-9) & (side == signal) & (sigma_d > 0)
feats = {"trend strength |gap|/vol": np.abs(gap) / sigma_d, "run since cross (vol units)": run / sigma_d,
         "age since cross (days)": age_d, "daily vol (%)": sigma_d * 100}
x = np.abs(exposure)
print(f"hours in position with the signal: {ok.sum()}  (against it: {((h >= start) & (np.abs(pos) > 1e-9) & (side != signal)).sum()}, "
      f"flat: {((h >= start) & (np.abs(pos) <= 1e-9)).sum()})")
for label, f in feats.items():
    q = np.quantile(f[ok], np.linspace(0, 1, 6))
    cells = []
    for lo, hi in zip(q, q[1:]):
        m = ok & (f >= lo) & (f <= hi)
        cells.append(f"{lo:5.1f}-{hi:5.1f}: {np.median(x[m]):.2f}x")
    print(f"  median exposure by {label:28s} | " + " | ".join(cells))

X = np.column_stack([np.log(feats["daily vol (%)"][ok]), feats["trend strength |gap|/vol"][ok], feats["run since cross (vol units)"][ok],
                     feats["age since cross (days)"][ok], np.ones(ok.sum())])
y = np.log(np.maximum(x[ok], 1e-4))
beta, *_ = np.linalg.lstsq(X, y, rcond=None)
pred = X @ beta
print(f"  log exposure regression R2 {1 - np.var(y - pred) / np.var(y):.2f}; coefs: log vol {beta[0]:+.2f}, strength {beta[1]:+.3f}, run {beta[2]:+.3f}, age {beta[3]:+.3f}")

# Orders: fills grouped by order id
fills = json.load(open(f"data/{name}/fills_ordered.json"))
orders = defaultdict(lambda: dict(sz=0.0, notional=0.0))
for f in fills:
    o = orders[f["oid"]]
    o.update(time=o.get("time", f["time"]), side=f["side"], dir=f["dir"], maker=not f["crossed"], start=o.get("start", float(f["startPosition"])))
    o["sz"] += float(f["sz"])
    o["notional"] += float(f["sz"]) * float(f["px"])
ords = sorted(orders.values(), key=lambda o: o["time"])
ords = [o for o in ords if o["time"] >= start]
for o in ords:
    i = bisect.bisect_right(h, o["time"]) - 1
    o.update(i=i, px=o["notional"] / o["sz"], av=av[i], kind=("add" if o["dir"].startswith("Open") else
                                                                 "flip" if ">" in o["dir"] else
                                                                 "exit" if abs(o["start"] + o["sz"] * (1 if o["side"] == "B" else -1)) < 1e-9 else "trim"))
print("\norders since Sep 2025:", {k: sum(o["kind"] == k for o in ords) for k in ("add", "trim", "exit", "flip")})
for kind in ("add", "trim", "exit", "flip"):
    sel = [o for o in ords if o["kind"] == kind]
    if not sel:
        continue
    pct = np.array([o["notional"] / o["av"] * 100 for o in sel])
    maker = np.mean([o["maker"] for o in sel])
    i = np.array([o["i"] for o in sel])
    pside = np.array([np.sign(o["start"]) if o["kind"] != "add" or abs(o["start"]) > 1e-9 else (1 if o["side"] == "B" else -1) for o in sel])
    print(f"  {kind:4s}: n {len(sel):4d}, maker {maker*100:3.0f}%, size {np.median(pct):.1f}% of account (p90 {np.percentile(pct,90):.1f}%), "
          f"at signal-aligned {np.mean(pside == signal[i])*100:3.0f}%, run since cross {np.median((run/sigma_d)[i]):+.1f} vol, "
          f"open position {np.median(np.abs(exposure[i])):.2f}x")

adds = [o for o in ords if o["kind"] == "add"]
steps = []
for a, b in zip(adds, adds[1:]):
    if a["side"] == b["side"] and b["time"] - a["time"] < 7 * 86400000:
        steps.append((b["px"] / a["px"] - 1) * (1 if b["side"] == "B" else -1))
steps = np.array(steps) * 100
print(f"  price step between consecutive same-side adds (in the trade's favour, %): median {np.median(steps):+.2f}, "
      f"quartiles {np.percentile(steps,25):+.2f} / {np.percentile(steps,75):+.2f}, share adding in favour {np.mean(steps > 0)*100:.0f}%")

# Exits against the cross: hours from the latest cross to each exit or flip
lags = []
for o in ords:
    if o["kind"] in ("exit", "flip"):
        i = o["i"]
        was = np.sign(o["start"])
        lags.append((i - cross[i]) / 24 if signal[i] != was else -1.0)
lags = np.array(lags)
print(f"\nexits/flips: {len(lags)}; after the signal turned against the position: {np.mean(lags >= 0)*100:.0f}% "
      f"(median {np.median(lags[lags >= 0]):.1f} days after the cross); while the signal still agreed: {np.mean(lags < 0)*100:.0f}%")

# Inner loop: what the last hours did, and how stretched price is from the fast EMA, at adds vs trims
fast = ema(mark, FAST)
stretch = (mark / fast - 1) * signal / np.where(sigma_d > 0, sigma_d, np.nan)
print("\ncontext at adds vs trims (in the trade's favour; vol = daily vol):")
for kind in ("add", "trim"):
    sel = [o for o in ords if o["kind"] == kind and o["i"] >= 24]
    i = np.array([o["i"] for o in sel])
    pside = signal[i]
    move = {n: (mark[i] / mark[i - n] - 1) * pside / sigma_d[i] for n in (4, 12, 24)}
    print(f"  {kind:4s}: last 4h {np.nanmedian(move[4]):+.2f} vol, 12h {np.nanmedian(move[12]):+.2f}, 24h {np.nanmedian(move[24]):+.2f} | "
          f"stretch from EMA{FAST}h {np.nanmedian(stretch[i]):+.2f} vol (quartiles {np.nanpercentile(stretch[i],25):+.2f} / {np.nanpercentile(stretch[i],75):+.2f})")

# Rebalancing to a target? Change in trend strength and in volatility over the hours before adds vs trims
strength = np.abs(gap) / np.where(sigma_d > 0, sigma_d, np.nan)
print("\nchange before adds vs trims (positive = rising):")
for kind in ("add", "trim"):
    sel = [o for o in ords if o["kind"] == kind and o["i"] >= 48]
    i = np.array([o["i"] for o in sel])
    for n in (6, 24):
        ds = strength[i] - strength[i - n]
        dv = np.log(sigma_d[i] / sigma_d[i - n])
        print(f"  {kind:4s} over {n:2d}h: strength {np.nanmedian(ds):+.3f} (rising {np.nanmean(ds > 0)*100:3.0f}%), vol {np.nanmedian(dv)*100:+.1f}% (rising {np.nanmean(dv > 0)*100:3.0f}%)")
rand = np.nonzero(ok)[0]
rand = rand[rand >= 48]
for n in (6, 24):
    ds = strength[rand] - strength[rand - n]
    print(f"  any held hour over {n:2d}h: strength rising {np.nanmean(ds > 0)*100:3.0f}%")

# Execution layer: is each buy below the previous order's price and each sell above it (buying dips, selling pops)?
print("\norder-to-order price moves (sub-hour execution):")
for kind in ("add", "trim"):
    better, gaps_min = [], []
    for prev, o in zip(ords, ords[1:]):
        if o["kind"] != kind:
            continue
        buy = o["side"] == "B"
        better.append((o["px"] < prev["px"]) if buy else (o["px"] > prev["px"]))
        gaps_min.append((o["time"] - prev["time"]) / 60000)
    print(f"  {kind:4s}: filled at a better price than the previous order {np.mean(better)*100:3.0f}%; "
          f"minutes since the previous order: median {np.median(gaps_min):.0f}")
pairs = [(a, b) for a, b in zip(ords, ords[1:]) if a["kind"] in ("add", "trim") and b["kind"] in ("add", "trim") and a["side"] != b["side"]]
edge = [((b["px"] / a["px"] - 1) * (1 if a["side"] == "B" else -1)) * 1e4 for a, b in pairs]
print(f"  opposite-side order pairs (buy then sell or sell then buy): {len(pairs)}, median spread captured {np.median(edge):+.0f} bps, "
      f"positive {np.mean(np.array(edge) > 0)*100:.0f}%")
