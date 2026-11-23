"""Backtest of Astraeus's rule as our own strategy, as pre-registered in FINDINGS.md (Backtest). `seen` checks the engine
on the in-sample period and touches no holdout night; `holdout` is the single holdout read and refuses a second
run. Read-only.

    python backtest.py seen
    python backtest.py holdout
"""
import datetime as dt
import json
import os
import sys

import numpy as np

import rule
from rule import DAY, HOUR, STEP, daily_participation, hourly, on_clock, participation
from stats import alpha, deflated_sharpe

FIT = ("2025-08-24", "2026-02-28")
SEEN = ("2025-08-24", "2026-11-19")
READS = {"binance": dict(window=("2017-11-01", "2025-08-23"), volume="spot", funding="binance", fee=0.0005),
         "hl": dict(window=("2024-02-01", "2025-08-23"), volume="hl", funding="hl", fee=0.00045)}
STRESS = 0.0010
FAMILY = [(48, 480), (36, 480), (60, 480), (48, 360), (48, 600)]
TRIALS = 250
REGIMES = [("2017-11-01", "2018-12-31", "2017-18 top and bear"), ("2019-01-01", "2019-12-31", "2019"),
           ("2020-01-01", "2021-12-31", "2020-21 bull"), ("2022-01-01", "2022-12-31", "2022 bear"),
           ("2023-01-01", "2024-12-31", "2023-24 bull"), ("2025-01-01", "2025-08-23", "2025 before the vault"),
           ("2025-08-24", "2026-11-19", "seen, in-sample")]
GUARD = "data/convexity/backtest_holdout.out"


def ms(day):
    return int(np.datetime64(day, "ms").astype(np.int64))


def simulate(aim, day_return, funding, fee, step=STEP):
    """Daily net returns and leverage of the nightly book. Each night it moves `step` of the way from yesterday's
    leverage, drifted by the day's move, to the aim, pays `fee` per unit of leverage traded, then holds a day:
    price move and funding (a positive rate charges longs). A day that would take equity below zero returns -1
    and the book stays flat after."""
    held, returns, leverage = 0.0, np.zeros(len(aim)), np.zeros(len(aim))
    for n in range(len(aim)):
        target = held + step * (aim[n] - held)
        gain = target * (day_return[n] - funding[n]) - fee * abs(target - held)
        if gain <= -1:
            returns[n] = -1.0
            break
        returns[n], leverage[n] = gain, target
        held = target * (1 + day_return[n]) / (1 + gain)
    return returns, leverage


class Market:
    """Per night d (00:00 UTC): the rule's inputs at the 01:00 close, the return from the 02:00 fill to the next
    night's and the funding paid in between."""

    def __init__(self):
        opens, self.close, spot_volume, self.missing = hourly("data/btc/spot_1h.json")
        perp_opens, _, perp_volume, _ = hourly("data/btc/perp_1h.json")
        self.days = np.arange(-(-opens[0] // DAY) * DAY, opens[-1] - 25 * HOUR + 1, DAY)
        self.signal = (self.days - opens[0]) // HOUR
        self.fill = self.signal + 1
        self.day_return = self.close[self.fill + 24] / self.close[self.fill] - 1
        self.variance = rule.variance(self.close)[self.signal]
        hl = json.load(open("data/btc/hl_1d.json"))
        hl_day = np.searchsorted(np.array([c["t"] for c in hl]) + DAY, self.days, side="right") - 1
        hl_ratio = daily_participation(np.array([float(c["v"]) for c in hl]))
        self.volume = {"spot": participation(spot_volume)[self.signal],
                       "perp": participation(on_clock(opens, perp_opens, perp_volume))[self.signal],
                       "hl": np.where(hl_day >= 0, hl_ratio[hl_day], np.nan)}
        binance = json.load(open("data/btc/funding.json"))
        hl_funding = json.load(open("data/btc/hl_funding.json"))
        self.funding = {"binance": self.day_funding(opens, [f["fundingTime"] for f in binance],
                                                    [float(f["fundingRate"]) for f in binance]),
                        "hl": self.day_funding(opens, [f["time"] for f in hl_funding],
                                               [float(f["fundingRate"]) for f in hl_funding])}
        month_ago = np.where(self.signal >= 720, self.signal - 720, 0)
        self.momentum = np.where(self.signal >= 720, np.sign(self.close[self.signal] / self.close[month_ago] - 1), np.nan)
        self.trends = {}

    def day_funding(self, opens, times, rates):
        by_hour = np.zeros(len(opens))
        slot = (np.asarray(times) - opens[0]) // HOUR
        inside = (slot >= 0) & (slot < len(opens))
        np.add.at(by_hour, slot[inside], np.asarray(rates)[inside])
        total = np.concatenate([[0.0], np.cumsum(by_hour)])
        return total[self.signal + 26] - total[self.signal + 2]

    def x(self, volume, fast=48, slow=480):
        if (fast, slow) not in self.trends:
            self.trends[fast, slow] = rule.trend(self.close, fast, slow)[self.signal]
        return self.trends[fast, slow] * self.volume[volume] / self.variance

    def between(self, first, last):
        return (self.days >= ms(first)) & (self.days <= ms(last))

    def worst_intraday(self, mask, leverage):
        """Worst move against the position between fills, times leverage."""
        hours = self.close[self.fill[mask][:, None] + np.arange(1, 25)] / self.close[self.fill[mask]][:, None] - 1
        hit = (leverage[:, None] * hours).min(axis=1)
        return hit.min(), self.days[mask][hit.argmin()]


def scales(market):
    """c per volume source and family member: least squares of the vault's aimed size on the rule's input over the
    first seen nights, as astraeus_test.py fits it."""
    nights = json.load(open("data/convexity/nightly.json"))
    day = np.array([n["t"] for n in nights]) - 105 * 60_000
    fit = (day >= ms(FIT[0])) & (day <= ms(FIT[1]))
    index = (day[fit] - market.days[0]) // DAY
    aimed = np.array([(n["y"] - (1 - STEP) * n["prev"]) / STEP for n in nights])[fit]
    return {(volume, member): (aimed * market.x(volume, *member)[index]).sum() / (market.x(volume, *member)[index] ** 2).sum()
            for volume in market.volume for member in FAMILY}


def compounded(returns):
    return np.prod(1 + returns) - 1


def summary(returns):
    equity = np.cumprod(1 + returns)
    return dict(total=compounded(returns), cagr=(1 + compounded(returns)) ** (365 / len(returns)) - 1,
                vol=returns.std() * np.sqrt(365), sharpe=returns.mean() / returns.std() * np.sqrt(365),
                drawdown=(equity / np.maximum.accumulate(equity) - 1).min())


def line(label, s):
    return (f"{label:34} total {s['total'] * 100:+8.1f}%  CAGR {s['cagr'] * 100:+6.1f}%  vol {s['vol'] * 100:5.1f}%  "
            f"Sharpe {s['sharpe']:5.2f}  max DD {s['drawdown'] * 100:6.1f}%")


def periods(days, length):
    stamp = [dt.datetime.fromtimestamp(d / 1000, dt.timezone.utc) for d in days]
    return np.array([f"{s.year}-{s.month if length == 'month' else (1 if s.month <= 6 else 2)}" for s in stamp])


def run(market, c, mask, volume, funding, fee, member=(48, 480)):
    aim = c[volume, member] * market.x(volume, *member)
    return simulate(aim[mask], market.day_return[mask], market.funding[funding][mask], fee)


def read(market, c, name, say):
    """One holdout read: the gates the pre-registration names, then what it reports."""
    spec = READS[name]
    mask = market.between(*spec["window"])
    days = market.days[mask]
    base, leverage = run(market, c, mask, spec["volume"], spec["funding"], spec["fee"])
    stress, _ = run(market, c, mask, spec["volume"], spec["funding"], STRESS)
    say(f"\n== {name} read: {len(days)} nights {spec['window'][0]} to {spec['window'][1]}, {spec['volume']} volume, "
        f"{spec['funding']} funding, {spec['fee'] * 1e4:.1f} bps a side")
    say(line("rule, base costs", summary(base)))
    say(line("rule, 10 bps a side", summary(stress)))
    s = summary(base)
    months = periods(days, "month")
    month_total = {m: compounded(base[months == m]) for m in np.unique(months)}
    gates = {"G2 profitable at 10 bps": compounded(stress) > 0,
             "G5 drawdown within 2.5 x vol": -s["drawdown"] <= 2.5 * s["vol"],
             "G7 without the 5 best days": compounded(np.sort(base)[:-5]) > 0,
             "G7 without the best month": compounded(base[months != max(month_total, key=month_total.get)]) > 0}
    if name == "binance":
        hold, _ = simulate(np.ones(mask.sum()), market.day_return[mask], market.funding["binance"][mask], spec["fee"], 1.0)
        momentum, _ = simulate(market.momentum[mask], market.day_return[mask], market.funding["binance"][mask], spec["fee"], 1.0)
        say(line("hold BTC 1x (B)", summary(hold)))
        say(line("30-day momentum 1x (M)", summary(momentum)))
        excess, exposures = alpha(base, [hold, momentum])
        measured = {n: deflated_sharpe(excess, n) for n in (5, TRIALS, 1000)}
        say(f"alpha vs B and M: exposures {exposures[0]:+.2f} / {exposures[1]:+.2f}, Sharpe {measured[TRIALS]['sharpe']:.2f}, "
            f"effective sample {measured[TRIALS]['effective']:.0f} days; DSR " +
            ", ".join(f"{measured[n]['dsr']:.2f} at N {n} (luck bar {measured[n]['luck']:.2f})" for n in measured))
        halves = periods(days, "half")
        full = [h for h in np.unique(halves) if (halves == h).sum() >= 181]
        positive = sum(compounded(base[halves == h]) > 0 for h in full)
        say(f"half-years profitable: {positive} of {len(full)}")
        family = {member: compounded(run(market, c, mask, "spot", "binance", spec["fee"], member)[0]) for member in FAMILY}
        say("family at base costs: " + ", ".join(f"{f}/{s_}h {v * 100:+.0f}%" for (f, s_), v in family.items()))
        gates = {"G1 DSR >= 0.80 at N 250": measured[TRIALS]["dsr"] >= 0.80, **gates,
                 "G3 >= 55% of half-years profitable": positive >= 0.55 * len(full),
                 "G8 >= 3 of 5 family members profitable": sum(v > 0 for v in family.values()) >= 3}
    else:
        check, _ = run(market, c, mask, "perp", "hl", spec["fee"])
        say(line("check: Binance perp volume", summary(check)))
    absolute = np.abs(leverage[leverage != 0])
    worst, worst_day = market.worst_intraday(mask, leverage)
    say(f"leverage: median {np.median(absolute):.2f}x, p99 {np.percentile(absolute, 99):.2f}x, max {absolute.max():.2f}x, "
        f"nights above 3x {(absolute > 3).sum()}; worst move against the book between fills x leverage "
        f"{worst * 100:.1f}% ({dt.date.fromtimestamp(worst_day / 1000)})")
    for gate, passed in gates.items():
        say(f"  {'PASS' if passed else 'FAIL'}  {gate}")
    return gates


def regimes(market, c, say):
    """P&L by regime on Binance inputs, one continuous book from the read's first night; the last row is in-sample."""
    mask = market.between(REGIMES[0][0], REGIMES[-1][1])
    returns, leverage = run(market, c, mask, "spot", "binance", READS["binance"]["fee"])
    days = market.days[mask]
    say("\n== regimes, Binance inputs, base costs (sum of daily returns; long and short days)")
    for first, last, label in REGIMES:
        part = (days >= ms(first)) & (days <= ms(last))
        r, lev = returns[part], leverage[part]
        say(f"{label:24} compounded {compounded(r) * 100:+7.1f}%  Sharpe {r.mean() / r.std() * np.sqrt(365):5.2f}  "
            f"long {r[lev > 0].sum() * 100:+6.1f}%  short {r[lev < 0].sum() * 100:+6.1f}%  BTC {compounded(market.day_return[mask][part]) * 100:+6.0f}%")


def seen(market, c):
    """Engine check against astraeus_test.py's free-running replay, then the seen period on both reads' inputs."""
    nights = json.load(open("data/convexity/nightly.json"))
    day = np.array([n["t"] for n in nights]) - 105 * 60_000
    px = np.array([n["px"] for n in nights])
    keep = np.flatnonzero((day >= ms(SEEN[0])) & (day <= ms(SEEN[1])))
    aim = c["perp", (48, 480)] * market.x("perp")[(day[keep] - market.days[0]) // DAY]
    held, replay = 0.0, np.zeros(len(keep))
    for n, a in enumerate(aim):
        held += STEP * (a - held)
        replay[n] = held * (px[keep[n] + 1] / px[keep[n]] - 1)
    mask = market.between(*SEEN)
    engine, _ = simulate(aim, market.day_return[mask], np.zeros(mask.sum()), 0.0)
    test = market.days[mask] >= ms("2026-03-01")
    print(f"engine check, perp volume, no costs, his test nights: replay {replay[test].sum() * 100:+.1f}%, engine "
          f"{engine[test].sum() * 100:+.1f}%, daily correlation {np.corrcoef(replay[test], engine[test])[0, 1]:.3f}")
    print(f"c: " + ", ".join(f"{v}/{f}-{s} {c[v, (f, s)]:.3g}" for (v, (f, s)) in c if (f, s) == (48, 480)))
    print(f"missing hourly bars filled: {market.missing}")
    for name, spec in READS.items():
        for fee in (spec["fee"], STRESS):
            returns, _ = run(market, c, mask, spec["volume"], spec["funding"], fee)
            print(line(f"seen, {name} inputs, {fee * 1e4:.1f} bps", summary(returns)))


if __name__ == "__main__":
    market = Market()
    c = scales(market)
    if sys.argv[1] == "seen":
        seen(market, c)
    else:
        if os.path.exists(GUARD):
            raise SystemExit(f"{GUARD} exists: the holdout was read. A rerun needs a bug fix logged with the results.")
        lines = [f"holdout read started {dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC"]
        open(GUARD, "w").write(lines[0] + "\n")

        def say(text):
            print(text)
            lines.append(text)

        gates = {f"{name} {gate}": passed for name in READS for gate, passed in read(market, c, name, say).items()}
        regimes(market, c, say)
        failed = [gate for gate, passed in gates.items() if not passed]
        say("\n== verdict: " + ("works (paper, holdout)" if not failed else "dead; failed: " + "; ".join(failed)))
        open(GUARD, "w").write("\n".join(lines) + "\n")
