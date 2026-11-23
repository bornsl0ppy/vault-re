"""Test Astraeus's +convexity rule on the nightly table, as pre-registered in FINDINGS.md
(Testing Astraeus's claims).
Needs nightly.py's table and data/btc/ (pull.py klines, funding, candles). Read-only.

    python astraeus_test.py convexity
"""
import json
import sys

import numpy as np
from scipy.stats import fisher_exact

import rule
from rule import DAY, HOUR, STEP, daily_participation, ema, hourly, on_clock, participation

name = sys.argv[1]


def ms(day):
    return int(np.datetime64(day, "ms").astype(np.int64))


def r2(actual, predicted):
    return 1 - ((actual - predicted) ** 2).sum() / ((actual - actual.mean()) ** 2).sum()


nights = json.load(open(f"data/{name}/nightly.json"))
t = np.array([n["t"] for n in nights])
prev = np.array([n["prev"] for n in nights])
y = np.array([n["y"] for n in nights])
order_night = np.array([n["order"] != 0 for n in nights])
px = np.array([n["px"] for n in nights])
r = np.append(px[1:] / px[:-1] - 1, np.nan)                         # 01:45 to the next night's 01:45
pnl, av = np.array([n["pnl"] for n in nights]), np.array([n["av"] for n in nights])
vault_actual = np.append(np.diff(pnl) / av[:-1], np.nan)
aimed = (y - (1 - STEP) * prev) / STEP                              # the size the vault aimed for before its step

# hourly inputs: the last bar closed by 01:45 is the 01:00 close
opens, close, spot_volume, _ = hourly("data/btc/spot_1h.json")
perp_opens, _, perp_volume, _ = hourly("data/btc/perp_1h.json")
bar_close = opens + HOUR
last = np.searchsorted(bar_close, t, side="right") - 1
trend, variance = rule.trend(close)[last], rule.variance(close)[last]
volume = participation(on_clock(opens, perp_opens, perp_volume))[last]
volume_spot = participation(spot_volume)[last]
hl = json.load(open("data/btc/hl_1d.json"))
hl_close_t = np.array([c["t"] for c in hl]) + DAY
volume_hl = daily_participation(np.array([float(c["v"]) for c in hl]))[np.searchsorted(hl_close_t, t, side="right") - 1]

# ours: daily 00:00 closes
is_midnight = bar_close % DAY == 0
day_close = close[is_midnight]
day_last = np.searchsorted(bar_close[is_midnight], t, side="right") - 1
day_returns = np.diff(np.log(day_close), prepend=np.nan)
vol20 = np.array([np.std(day_returns[k - 19:k + 1]) if k >= 20 else np.nan for k in range(len(day_close))])
z = ((ema(day_close, 3) / ema(day_close, 14) - 1) / vol20)[day_last]

funding = json.load(open("data/btc/funding.json"))
funding_t, funding_rate = np.array([f["fundingTime"] for f in funding]), np.array([float(f["fundingRate"]) for f in funding])
midnight = t - 105 * 60_000
prior_funding = np.array([funding_rate[(funding_t > m - 16.5 * HOUR) & (funding_t <= m + 0.5 * HOUR)].mean()
                          for m in midnight])                       # 08:00 and 16:00 the day before, 00:00
assert np.isfinite(np.c_[trend, variance, volume, volume_spot, volume_hl, z, prior_funding]).all()

FEATURES = {"A s/sigma": trend / np.sqrt(variance), "B s/sigma2": trend / variance,
            "C s.v/sigma": trend * volume / np.sqrt(variance), "D s.v/sigma2 (his)": trend * volume / variance}
CHECKS = {"D, spot volume": trend * volume_spot / variance, "D, HL daily volume": trend * volume_hl / variance}


def fit_scale(x, fit):
    return (aimed[fit] * x[fit]).sum() / (x[fit] ** 2).sum()


def fit_ours(fit):
    k, cap = np.meshgrid(np.arange(0.02, 1.5, 0.01), np.arange(0.2, 2.5, 0.05), indexing="ij")
    err = ((aimed[fit][None, None, :] - np.clip(k[..., None] * z[fit][None, None, :], -cap[..., None], cap[..., None])) ** 2).sum(-1)
    a, b = np.unravel_index(err.argmin(), err.shape)
    return k[a, b], cap[a, b]


def free_run(aim):
    held, out = 0.0, np.empty(len(aim))
    for n, a in enumerate(aim):
        held += STEP * (a - held)
        out[n] = held
    return out


def scores(aim, test):
    one_step = prev + STEP * (aim - prev)
    held = free_run(aim)
    live = test & np.isfinite(r)
    big = test & (np.abs(y) > 1)
    model_pnl, vault_pnl = held * r, y * r
    return dict(level=r2(y[test], one_step[test]), order=r2((y - prev)[test], (one_step - prev)[test]),
                direction=np.mean(np.sign(one_step - prev)[test & order_night] == np.sign(y - prev)[test & order_night]),
                pnl_share=model_pnl[live].sum() / vault_pnl[live].sum(),
                sharpe=model_pnl[live].mean() / model_pnl[live].std() * np.sqrt(365),
                corr=np.corrcoef(model_pnl[live], vault_pnl[live])[0, 1],
                big_one_step=(one_step[big] * np.sign(y[big])).sum() / np.abs(y[big]).sum(),
                big_free=(held[big] * np.sign(y[big])).sum() / np.abs(y[big]).sum())


def bootstrap(aims, test, draws=2000, block=10):
    idx = np.flatnonzero(test)
    starts = np.random.default_rng(0).integers(0, len(idx) - block + 1, size=(draws, -(-len(idx) // block)))
    samples = idx[(starts[..., None] + np.arange(block)).reshape(draws, -1)[:, :len(idx)]]
    out = {}
    for label, aim in aims.items():
        step = STEP * (aim - prev)
        actual = y - prev
        out[label] = np.array([r2(actual[s], step[s]) for s in samples])
    return out


def mixture(x, iters=1000):
    """Two-component Gaussian mixture by EM; component 0 is the lower ("off") one."""
    weight, mu, sd = 0.5, np.percentile(x, [20, 80]), np.full(2, x.std())
    for _ in range(iters):
        dens = np.array([weight, 1 - weight])[:, None] * np.exp(-0.5 * ((x - mu[:, None]) / sd[:, None]) ** 2) / (sd[:, None] * np.sqrt(2 * np.pi))
        resp = dens / dens.sum(0)
        weight = resp[0].mean()
        mu = (resp * x).sum(1) / resp.sum(1)
        sd = np.sqrt((resp * (x - mu[:, None]) ** 2).sum(1) / resp.sum(1))
    if mu[0] > mu[1]:
        weight, mu, sd, resp = 1 - weight, mu[::-1], sd[::-1], resp[::-1]
    return weight, mu, sd, resp, np.log(dens.sum(0)).sum()


day = np.array([n["day"] for n in nights])
SPLITS = {"his": t < ms("2026-03-01"), "halves": np.arange(len(t)) < len(t) // 2}
results, bars = {}, {}
for split, fit in SPLITS.items():
    test = ~fit
    aims = {label: fit_scale(x, fit) * x for label, x in FEATURES.items()}
    k, cap = fit_ours(fit)
    aims["E ours"] = np.clip(k * z, -cap, cap)
    checks = {label: fit_scale(x, fit) * x for label, x in CHECKS.items()}
    switch = aims["D s.v/sigma2 (his)"] * (prior_funding >= 0)
    live = test & np.isfinite(r)
    print(f"\n== split {split}: fit {day[fit][0]} to {day[fit][-1]} ({fit.sum()} nights), test {day[test][0]} to "
          f"{day[test][-1]} ({test.sum()}); E: k {k:.2f} cap {cap:.2f}")
    print(f"vault on test nights: replay sum y.r {np.nansum((y * r)[live]) * 100:+.1f}%, actual "
          f"{np.nansum(vault_actual[live]) * 100:+.1f}%, Sharpe replay {(y * r)[live].mean() / (y * r)[live].std() * np.sqrt(365):.2f}")
    print(f"{'model':22} {'level':>6} {'order':>6} {'dir':>5} {'pnl/vault':>9} {'sharpe':>6} {'corr':>5} {'big 1-step':>10} {'big free':>8}")
    print(f"{'P persistence':22} {r2(y[test], prev[test]):6.2f} {r2((y - prev)[test], 0 * prev[test]):6.2f}")
    results[split] = {}
    for label, aim in {**aims, **checks, "D + funding switch": switch}.items():
        s = scores(aim, test)
        results[split][label] = s
        print(f"{label:22} {s['level']:6.2f} {s['order']:6.2f} {s['direction']:5.0%} {s['pnl_share']:9.0%} "
              f"{s['sharpe']:6.2f} {s['corr']:5.2f} {s['big_one_step']:10.0%} {s['big_free']:8.0%}")
    boot = bootstrap({key: aims[key] for key in FEATURES}, test)
    A, B, C, D = (boot[key] for key in FEATURES)
    for label, diff in {"D-B volume": D - B, "D-C variance": D - C, "D-A both": D - A}.items():
        lo, hi = np.percentile(diff, [5, 95])
        print(f"bootstrap order-R2 {label}: {np.median(diff):+.3f} (90% {lo:+.3f} to {hi:+.3f})")
        bars.setdefault(label, {})[split] = lo
    if split == "his":
        his_aim_D = aims["D s.v/sigma2 (his)"]

# two-state, on all nights, scale from his split
m = aimed / his_aim_D
usable = (np.abs(his_aim_D) > 0.05) & (m > 0)
x = np.log(m[usable])
weight, mu, sd, resp, loglik = mixture(x)
loglik1 = (-0.5 * ((x - x.mean()) / x.std()) ** 2 - np.log(x.std() * np.sqrt(2 * np.pi))).sum()
delta_bic = (-2 * loglik1 + 2 * np.log(len(x))) - (-2 * loglik + 5 * np.log(len(x)))
off = np.zeros(len(t), bool)
off[usable] = resp[0] > 0.5
pairs = usable[1:] & usable[:-1]
after_off, after_on = pairs & off[:-1], pairs & ~off[:-1]
lag1 = np.corrcoef(np.log(m[1:][pairs]), np.log(m[:-1][pairs]))[0, 1]
excluded = (np.abs(his_aim_D) > 0.05) & (m <= 0)
print(f"\n== two-state on log m ({usable.sum()} nights; {excluded.sum()} more have m <= 0, {(excluded & (y == 0)).sum()} of them flat)")
print(f"delta BIC (1 vs 2 components) {delta_bic:.1f}; pi {weight:.2f}; e^mu off {np.exp(mu[0]):.2f}, on {np.exp(mu[1]):.2f}; "
      f"sd {sd[0]:.2f}/{sd[1]:.2f}")
print(f"P(off | off) {off[1:][after_off].mean():.0%} (n {after_off.sum()}), P(off | on) {off[1:][after_on].mean():.0%} "
      f"(n {after_on.sum()}); lag-1 autocorrelation of log m {lag1:.2f}")
negative = prior_funding < 0
table = [[(off & negative & usable).sum(), (off & ~negative & usable).sum()],
         [(~off & negative & usable).sum(), (~off & ~negative & usable).sum()]]
odds, p = fisher_exact(table, alternative="greater")
print(f"funding: off after negative {table[0][0]}/{table[0][0] + table[1][0]}, after non-negative "
      f"{table[0][1]}/{table[0][1] + table[1][1]}; odds ratio {odds:.2f}, one-sided p {p:.3f}")

D = "D s.v/sigma2 (his)"
verdicts = {
    "1 volume": all(results[s]["C s.v/sigma"]["order"] > results[s]["A s/sigma"]["order"] and
                    results[s][D]["order"] > results[s]["B s/sigma2"]["order"] and bars["D-B volume"][s] > 0 for s in SPLITS),
    "2 variance": all(results[s]["B s/sigma2"]["order"] > results[s]["A s/sigma"]["order"] and
                      results[s][D]["order"] > results[s]["C s.v/sigma"]["order"] and bars["D-C variance"][s] > 0 for s in SPLITS),
    "3 headline": results["his"][D]["level"] >= 0.65 and results["his"][D]["direction"] >= 0.72,
    "4 two-state": delta_bic > 10 and 0.10 <= weight <= 0.35 and np.exp(mu[0]) < 0.3 and 0.8 <= np.exp(mu[1]) <= 1.5,
    "5 funding switch": p < 0.05 and all(results[s]["D + funding switch"]["sharpe"] > results[s][D]["sharpe"] for s in SPLITS),
    "6 replicate (b)": all(results[s][D]["pnl_share"] >= 0.5 for s in SPLITS),
}
print("\n== bars: " + "; ".join(f"{k} {'PASS' if v else 'fail'}" for k, v in verdicts.items()))
