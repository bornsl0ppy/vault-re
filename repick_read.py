"""+convexity's frozen rule on a monthly re-picked list of the busiest alts, read once. The spec is in FINDINGS.md
(section 4). The read also judged the basket against our own live strategy; that comparison is private, and this copy
compares with holding BTC instead. Read-only.

    python repick_read.py check     # the lists over time, and BTC through the trading path against the engine
    python repick_read.py read      # the one read; refuses a rerun
"""
import datetime as dt
import sys
from pathlib import Path

import numpy as np

import backtest
import binance_archive
from backtest import STEP, compounded, line, ms, periods, run, summary
from coins_read import CoinMarket, holding, mixed, sharpe
from grid_backtest import hl_listings
from rule import DAY

SLOTS, KEEP, LISTED_DAYS, WINDOW_DAYS = 10, 15, 60, 30
HL_FROM = ms("2023-05-24")                     # Hyperliquid lists 20 perps besides BTC, ETH and SOL
GATED = ("2021-01-01", "2025-08-23")
SEEN = ("2025-08-24", "2026-09-30")
FEE, STRESS = 0.0005, 0.0010
TRIALS = 281
GUARD = Path("data/coins/repick.out")


def pick(ranking, current, slots=SLOTS, keep=KEEP):
    """This month's members: current members still ranked within `keep` stay, and free slots go to the
    highest-ranked others. `ranking` is best first and holds only eligible coins."""
    rank = {coin: i for i, coin in enumerate(ranking)}
    stay = {coin for coin in current if rank.get(coin, keep) < keep}
    return stay | set([coin for coin in ranking if coin not in stay][:slots - len(stay)])


def trade(aim, day_return, funding, fee, member, step=STEP):
    """One coin's nightly book: as backtest.simulate while a member; off the list a position can only shrink (it
    steps toward a smaller aim in its own direction, holds when the aim is as large or larger, and closes when the
    aim is zero or flips), and a flat coin off the list stays out."""
    held, returns, leverage = 0.0, np.zeros(len(aim)), np.zeros(len(aim))
    for n in range(len(aim)):
        if member[n]:
            target = held + step * (aim[n] - held)
        elif held == 0:
            continue
        elif aim[n] * held > 0:
            target = held + step * (aim[n] - held) if abs(aim[n]) < abs(held) else held
        else:
            target = 0.0
        gain = target * (day_return[n] - funding[n]) - fee * abs(target - held)
        if gain <= -1:
            returns[n] = -1.0
            break
        returns[n], leverage[n] = gain, target
        held = target * (1 + day_return[n]) / (1 + gain)
    return returns, leverage


def month(t):
    return dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m")


def lists():
    """Each month's members, re-picked at 00:00 UTC on its first day, from 2021-01 to 2026-09."""
    listed = hl_listings()
    volumes = {}
    for symbol in binance_archive.coins():
        if symbol == "BTCUSDT":
            continue
        bars = binance_archive.hourly(symbol)
        traded = np.flatnonzero(bars["quote_volume"] > 0)
        if len(traded):
            volumes[symbol] = (bars["open_time"], np.concatenate([[0.0], np.cumsum(bars["quote_volume"])]),
                               bars["open_time"][traded[0]], bars["open_time"][traded[-1]])
    out, current = {}, set()
    starts = [dt.datetime(y, m, 1, tzinfo=dt.timezone.utc) for y in range(2021, 2027) for m in range(1, 13)]
    for start in starts:
        t = int(start.timestamp() * 1000)
        if t > ms(SEEN[1]):
            break
        volume = {}
        for symbol, (times, total, first, last) in volumes.items():
            if first > t - LISTED_DAYS * DAY or last < t - DAY:
                continue
            if t >= HL_FROM and not (symbol in listed and listed[symbol][0] <= t <= listed[symbol][1]):
                continue
            a, b = np.searchsorted(times, [t - WINDOW_DAYS * DAY, t])
            volume[symbol] = total[b] - total[a]
        current = pick(sorted(volume, key=lambda s: -volume[s]), current)
        out[month(t)] = current
    return out


def basket(markets, c, members, first, last, fee, family=(48, 480)):
    """The basket's calendar nights and daily returns (a tenth of the sum over coins), and each coin's returns."""
    per_coin = {}
    for symbol, market in markets.items():
        mask = market.between(first, last) & np.isfinite(market.x("perp", *family)) & np.isfinite(market.day_return)
        days = market.days[mask]
        member = np.array([symbol in members.get(month(d), ()) for d in days])
        if not member.any():
            continue
        aim = c["perp", family] * market.x("perp", *family)[mask]
        returns, _ = trade(aim, market.day_return[mask], market.funding["binance"][mask], fee, member)
        per_coin[symbol] = dict(zip(days.tolist(), returns))
    days = np.arange(ms(first), ms(last) + 1, DAY)
    return days, np.array([sum(r.get(d, 0.0) for r in per_coin.values()) / SLOTS for d in days]), per_coin


def setup():
    btc = backtest.Market()
    members = lists()
    needed = set().union(*members.values()) | {"ETHUSDT", "SOLUSDT", "BTCUSDT"}
    return backtest.scales(btc), members, {s: CoinMarket(s) for s in sorted(needed)}


def describe(members):
    keys = sorted(members)
    changes = [len(members[b] - members[a]) for a, b in zip(keys, keys[1:])]
    coins = set().union(*members.values())
    return [f"{len(keys)} monthly lists {keys[0]} to {keys[-1]}, {len(coins)} coins ever members; replaced a month: "
            f"median {np.median(changes):.0f}, mean {np.mean(changes):.1f}, max {max(changes)}",
            *[f"  {k}: " + ", ".join(sorted(s.removesuffix("USDT") for s in members[k]))
              for k in ("2021-01", "2022-05", "2023-06", "2024-06", "2025-08", "2026-09") if k in members]]


def check():
    """The lists (volume only, no returns), and BTC through `trade` with every night a member against the engine."""
    c, members, markets = setup()
    print("\n".join(describe(members)))
    btc = markets["BTCUSDT"]
    mask = btc.between("2019-10-15", GATED[1]) & np.isfinite(btc.x("perp"))
    engine, _ = run(btc, c, mask, "perp", "binance", FEE)
    ours, _ = trade(c["perp", (48, 480)] * btc.x("perp")[mask], btc.day_return[mask], btc.funding["binance"][mask], FEE,
                    np.ones(mask.sum(), bool))
    print(f"BTC through trade() against the engine, {mask.sum()} nights: largest daily difference "
          f"{np.abs(engine - ours).max():.2e}")


def read():
    if GUARD.exists():
        raise SystemExit(f"{GUARD} exists: the read was done. A rerun needs a bug fix logged with the results.")
    GUARD.parent.mkdir(parents=True, exist_ok=True)
    GUARD.write_text("read started\n")
    lines = []

    def say(text):
        print(text)
        lines.append(text)

    c, members, markets = setup()
    days, base, per_coin = basket(markets, c, members, *GATED, FEE)
    _, stress, _ = basket(markets, c, members, *GATED, STRESS)
    btc = holding(markets, np.concatenate([days, [days[-1] + DAY]]))
    s = summary(base)
    say(f"== +convexity's rule on a monthly re-picked list of {SLOTS} alts, decisions {GATED[0]} to {GATED[1]} "
        f"({len(days)} nights), perp volume, Binance funding, {FEE * 1e4:.0f} bps a side")
    say("\n".join(describe(members)))
    say(line("basket, base costs", s))
    say(line("basket, 10 bps a side", summary(stress)))
    say(line("holding BTC", summary(btc)))
    say(f"correlation with holding BTC {np.corrcoef(base, btc)[0, 1]:+.2f}; Sharpe of a 50/50 mix at BTC's volatility "
        f"{mixed(base, btc):.2f} against BTC's {sharpe(btc):.2f}")
    months, halves = periods(days, "month"), periods(days, "half")
    month_total = {m: compounded(base[months == m]) for m in np.unique(months)}
    full = [h for h in np.unique(halves) if (halves == h).sum() >= 181]
    positive = sum(compounded(base[halves == h]) > 0 for h in full)
    say("half-years: " + ", ".join(f"{h} {compounded(base[halves == h]) * 100:+.0f}%" for h in full))
    family = {f: compounded(basket(markets, c, members, *GATED, FEE, f)[1]) for f in backtest.FAMILY}
    say("family at base costs: " + ", ".join(f"{f}/{w}h {v * 100:+.0f}%" for (f, w), v in family.items()))
    gates = {"G2 profitable at 10 bps": compounded(stress) > 0,
             "G3 >= 55% of half-years profitable": positive >= 0.55 * len(full),
             "G5 drawdown within 2.5 x vol": -s["drawdown"] <= 2.5 * s["vol"],
             "G7 without the 5 best days": compounded(np.sort(base)[:-5]) > 0,
             "G7 without the best month": compounded(base[months != max(month_total, key=month_total.get)]) > 0,
             "G8 >= 3 of 5 family members profitable": sum(v > 0 for v in family.values()) >= 3}
    for gate, passed in gates.items():
        say(f"  {'PASS' if passed else 'FAIL'}  {gate}")
    say("  (B, what the basket adds to our own live strategy, is judged privately)")
    say("\n== reported, not gated")
    share = {symbol: sum(r.values()) / SLOTS for symbol, r in per_coin.items()}
    ordered = sorted(share, key=share.get)
    say("P&L by coin, summed daily returns as a share of the basket: best " +
        ", ".join(f"{s.removesuffix('USDT')} {share[s] * 100:+.0f}%" for s in ordered[::-1][:6]) + "; worst " +
        ", ".join(f"{s.removesuffix('USDT')} {share[s] * 100:+.0f}%" for s in ordered[:6]))
    seen_days, seen, _ = basket(markets, c, members, *SEEN, FEE)
    say(line(f"seen {SEEN[0]} to {SEEN[1]}", summary(seen)))
    failed = [g for g, ok in gates.items() if not ok]
    say("\n== gates G2-G8: " + ("all pass" if not failed else "failed: " + "; ".join(failed)))
    GUARD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    {"check": check, "read": read}[sys.argv[1]]()
