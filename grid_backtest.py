"""Backtest of Cold Process's decoded DCA grid as our own strategy, on Binance minute paths through 2020-2025's
crashes, as pre-registered in FINDINGS.md (section 6). Read-only.

    python grid_backtest.py check      # engine check: the vault's own deal starts replayed on Binance
    python grid_backtest.py fit        # selection rules against the vault's own picks
    python grid_backtest.py seen       # the rule over the vault's own months, beside its record
    python grid_backtest.py holdout    # the one read, 2020-01-08 to 2026-01-31; refuses a rerun
"""
import bisect
import collections
import datetime as dt
import functools
import json
import pickle
import sys
from pathlib import Path

import numpy as np

import binance_archive
import stats

MINUTE, HOUR, DAY = 60_000, 3_600_000, 86_400_000
CLOCK0 = 1_567_296_000_000                      # 2019-09-01 00:00 UTC, before the first USDT perp alt

ENTRY = 0.0123                                  # of equity
FIRST_RUNG, RUNG_GROWTH = 1.33, 2.33            # first rung against the entry, each later one against the one before
DEPTHS = (0.026, 0.041, 0.059, 0.080, 0.110, 0.124, 0.149)
CAP = 1.25                                      # position cost over the equity at entry
TAKE_PROFIT = 0.0134                            # over the average entry
UNWIND_HOURS = 72
STOP = 0.10                                     # under the average entry, once capped
RESTART = 5 * MINUTE

Costs = collections.namedtuple("Costs", "taker maker")
BASE = Costs(taker=4.5e-4, maker=1.5e-4)
STRESS = Costs(taker=10e-4, maker=10e-4)
MISSING_FUNDING = 1e-4                          # per 8h, for a month without prints

NEVER = {"BTCUSDT", "ETHUSDT", "SOLUSDT"}
HL_FROM = 20                                    # HL perps listed (besides BTC, ETH, SOL) before its list is the universe
# rule: (pool size by 24h volume, ranked by, universe). "hl": every Binance coin until Hyperliquid lists HL_FROM
# perps, then only the coins it lists at that moment; "binance": every Binance coin throughout.
SELECTIONS = {"volatile": (10, "volatility 7d", "hl"), "week": (20, "move 7d", "hl"), "day": (20, "move 24h", "hl"),
              "binance": (10, "volatility 7d", "binance")}
FAMILY = {"centre": ("volatile", "unwind"), "hold": ("volatile", "hold"), "stop": ("volatile", "stop"),
          "week": ("week", "unwind"), "day": ("day", "unwind"), "binance-wide": ("binance", "unwind")}
TRIALS = 1_100


def ms(day):
    return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)


HOLDOUT = (ms("2020-01-08"), ms("2026-02-01"))
SEEN = (ms("2026-02-01"), ms("2026-11-01"))     # the archive's hourly files end with October
CRASHES = {"2021-05-19": ("2021-05-17", "2021-05-24"), "LUNA": ("2022-05-07", "2022-05-14"),
           "FTX": ("2022-11-06", "2022-11-13"), "2024-08-05": ("2024-08-03", "2024-08-10"),
           "2025-10-10": ("2025-10-09", "2025-10-16")}


def ladder(entry_price, entry_cost, cap_cost):
    """The rungs below an entry as (price, cost), the last one cut where the position would pass the cap."""
    out, cost, size = [], entry_cost, entry_cost * FIRST_RUNG
    for depth in DEPTHS:
        size_here = min(size, cap_cost - cost)
        if size_here <= 1e-12 * cap_cost:
            break
        out.append((entry_price * (1 - depth), size_here))
        cost += size_here
        size *= RUNG_GROWTH
    return out


def deal(bars, entry_time, entry_price, equity, deep, costs, prints=(), until=None):
    """One deal bought at entry_price, then run over `bars` ((open time, high, low, close) per minute, from the
    minute after the entry) until it exits. deep is what happens at the cap: "unwind", "hold" or "stop". prints:
    funding (time, rate). until: mark the deal to market there if it's still open.

    Returns the exit (take-profit, stop, unwound, delisted, ruin, open), its time, equity after it, rungs filled,
    peak cost over the equity at entry, fees, funding, the lowest equity on the way, the equity at each UTC
    midnight and the quantity still held."""
    cap_cost = CAP * equity
    entry_cost = ENTRY * equity
    rungs = ladder(entry_price, entry_cost, cap_cost)
    quantity, cost = entry_cost / entry_price, entry_cost
    fees = entry_cost * costs.taker
    cash = equity - entry_cost - fees
    paid, filled, capped, lowest, marks = 0.0, 0, None, equity, []
    prints = iter(prints)
    upcoming = next(prints, None)
    while upcoming is not None and upcoming[0] <= entry_time:
        upcoming = next(prints, None)
    end, close = entry_time + MINUTE, entry_price

    def out(kind, when):
        return {"exit": kind, "end": when, "equity": cash + quantity * close, "rungs": filled,
                "peak": cost_peak / equity, "fees": fees, "funding": paid, "lowest": lowest, "marks": marks,
                "quantity": quantity}

    def sell(amount, price, fee):
        nonlocal cash, fees, quantity, cost
        cash += amount * price * (1 - fee)
        fees += amount * price * fee
        cost -= cost / quantity * amount
        quantity -= amount

    cost_peak = cost
    for t, high, low, close_here in bars:
        if until is not None and t + MINUTE > until:
            return out("open", end)
        end, close = t + MINUTE, close_here
        while upcoming is not None and upcoming[0] <= end:
            charge = upcoming[1] * quantity * close
            cash -= charge
            paid += charge
            upcoming = next(prints, None)
        added = False
        while filled < len(rungs) and low < rungs[filled][0]:
            price, rung_cost = rungs[filled]
            quantity += rung_cost / price
            cost += rung_cost
            cash -= rung_cost * (1 + costs.maker)
            fees += rung_cost * costs.maker
            filled += 1
            added = True
        cost_peak = max(cost_peak, cost)
        if capped is None and cost >= cap_cost * (1 - 1e-9):
            capped = quantity
        lowest = min(lowest, cash + quantity * low)
        average = cost / quantity
        target = average * (1 + TAKE_PROFIT)
        if not added and high > target:
            sell(quantity, target, costs.maker)
            return out("take-profit", end)
        if capped is not None:
            if deep == "stop" and close < average * (1 - STOP):
                sell(quantity, close, costs.taker)
                return out("stop", end)
            if deep == "unwind" and end % HOUR == 0 and close < average:
                sell(min(quantity, capped / UNWIND_HOURS), close, costs.taker)
                if quantity <= 1e-9 * capped:
                    quantity = 0.0
                    return out("unwound", end)
        if cash + quantity * close <= 0:
            return out("ruin", end)
        if end % DAY == 0:
            marks.append((end, cash + quantity * close))
    if until is not None and end >= until:
        return out("open", end)
    sell(quantity, close, costs.taker)
    return out("delisted", end)


def month_of(t):
    return dt.datetime.fromtimestamp(t / 1000, dt.timezone.utc).strftime("%Y-%m")


def next_month(month):
    year, number = int(month[:4]), int(month[5:])
    return f"{year + number // 12}-{number % 12 + 1:02d}"


def binance_symbol(coin):
    return f"1000{coin[1:]}USDT" if coin.startswith("k") and coin[1:].isupper() else f"{coin}USDT"


def hl_listings():
    """When each Hyperliquid perp traded, first to last daily candle with trades (ms), keyed by its Binance symbol.
    HL serves empty candles from before a listing. Cached in data/binance_um/hl_listings.json."""
    path = binance_archive.ROOT / "hl_listings.json"
    if not path.exists():
        from pull import candles
        out = {}
        for row in json.load(open("data/hl_meta_universe.json")):
            days = [c for c in candles(row["name"], "1d", dt.datetime(2020, 1, 1, tzinfo=dt.timezone.utc)) if c["n"] > 0]
            if days:
                out[binance_symbol(row["name"])] = (days[0]["t"], days[-1]["t"] + DAY)
        json.dump(out, open(path, "w"), indent=1)
    return json.load(open(path))


def market():
    """Per hour from CLOCK0, each selection rule's candidates in order (coin indices, -1 padded) from the hourly
    bars closed by the end of that hour, each coin's first and last hour, and its price at every UTC midnight
    (carried past its last bar). Cached in data/binance_um/market.npz."""
    path = binance_archive.ROOT / "market.npz"
    if path.exists():
        return dict(np.load(path))
    names = binance_archive.coins()
    hours = (SEEN[1] - CLOCK0) // HOUR
    features = {key: np.full((hours, len(names)), np.nan, np.float32) for _, key, _ in SELECTIONS.values()}
    volume = np.full((hours, len(names)), np.nan, np.float32)
    midnight = np.full((hours // 24 + 1, len(names)), np.nan)
    first, last = np.full(len(names), -1), np.full(len(names), -1)
    for c, name in enumerate(names):
        bars = binance_archive.hourly(name)
        index = (bars["open_time"] - CLOCK0) // HOUR
        traded = np.flatnonzero(bars["quote_volume"] > 0)
        if len(traded) == 0:
            continue
        # a delisted contract's archive runs on with frozen bars (FTT at 1.59 since 2022-11): its life ends at the
        # last hour with volume
        keep = (index < hours) & (np.arange(len(index)) >= traded[0]) & (np.arange(len(index)) <= traded[-1])
        index, close_raw, quote_raw = index[keep], bars["close"][keep], bars["quote_volume"][keep]
        if len(index) < 170:
            continue
        first[c], last[c] = index[0], index[-1]
        n = index[-1] - index[0] + 1
        close = np.full(n, np.nan)
        close[index - index[0]] = close_raw
        filled = np.where(~np.isnan(close), np.arange(n), 0)
        close = close[np.maximum.accumulate(filled)]
        quote = np.zeros(n)
        quote[index - index[0]] = quote_raw
        sums = np.concatenate([[0.0], np.cumsum(quote)])
        logs = np.diff(np.log(close))
        log_sums = np.concatenate([[0.0], np.cumsum(logs)])
        square_sums = np.concatenate([[0.0], np.cumsum(logs ** 2)])
        k = np.arange(168, n)
        rows = index[0] + k
        volume[rows, c] = sums[k + 1] - sums[k - 23]
        mean = (log_sums[k] - log_sums[k - 168]) / 168
        features["volatility 7d"][rows, c] = np.sqrt(np.maximum((square_sums[k] - square_sums[k - 168]) / 168 - mean ** 2, 0) * 168 / 167)
        features["move 24h"][rows, c] = close[k] / close[k - 24] - 1
        features["move 7d"][rows, c] = close[k] / close[k - 168] - 1
        days = np.arange((index[0] + 1 + 23) // 24, midnight.shape[0])
        at = np.minimum(days * 24 - 1 - index[0], n - 1)
        midnight[days, c] = close[at]
    allowed = np.array([name not in NEVER for name in names])
    listed = hl_listings()
    hl_first = np.array([(listed[n][0] - CLOCK0) // HOUR if n in listed else hours for n in names])
    hl_last = np.array([(listed[n][1] - CLOCK0) // HOUR if n in listed else -1 for n in names])
    on_hl = lambda h: (hl_first <= h) & (hl_last >= h) & allowed
    switch = next(h for h in range(hours) if on_hl(h).sum() >= HL_FROM)
    out = {rule: np.full((hours, n), -1, np.int16) for rule, (n, _, _) in SELECTIONS.items()}
    for h in range(hours):
        trading = ~np.isnan(volume[h]) & allowed
        for universe in ("hl", "binance"):
            eligible = np.flatnonzero(trading & on_hl(h) if universe == "hl" and h >= switch else trading)
            by_volume = eligible[np.argsort(-volume[h, eligible], kind="stable")]
            for rule, (n, key, rule_universe) in SELECTIONS.items():
                if rule_universe == universe:
                    top = by_volume[:n]
                    out[rule][h, :len(top)] = top[np.argsort(-features[key][h, top], kind="stable")]
    out.update(names=np.array(names), first=first, last=last, midnight=midnight, switch=switch)
    np.savez(path, **out)
    return out


@functools.lru_cache(maxsize=16)
def month_bars(symbol, month):
    bars = binance_archive.minutes(symbol, month)
    return bars["open_time"].tolist(), bars["high"].tolist(), bars["low"].tolist(), bars["close"].tolist()


def minutes_from(symbol, t, end):
    """The coin's minute bars opening from t until end, month by month."""
    month = month_of(t)
    while month <= month_of(end - 1):
        times, highs, lows, closes = month_bars(symbol, month)
        i, j = bisect.bisect_left(times, t), bisect.bisect_left(times, end)
        yield from zip(times[i:j], highs[i:j], lows[i:j], closes[i:j])
        month = next_month(month)


@functools.lru_cache(maxsize=64)
def funding_prints(symbol, first, last):
    """The coin's funding prints between its first and last hour; a month without any gets MISSING_FUNDING every
    8h. Returns times, rates and how many prints were made up."""
    prints = binance_archive.funding(symbol)
    times, rates = prints["time"].tolist(), prints["rate"].tolist()
    have = {month_of(t) for t in times}
    start = CLOCK0 + first * HOUR
    made_up = [(t, MISSING_FUNDING) for t in range(-(-start // (8 * HOUR)) * 8 * HOUR, CLOCK0 + (last + 1) * HOUR, 8 * HOUR)
               if month_of(t) not in have]
    merged = sorted(list(zip(times, rates)) + made_up)
    return [t for t, _ in merged], [r for _, r in merged], len(made_up)


def open_deal(symbol, c, m, t, equity, deep, costs, until):
    """A deal on the coin bought at the close of the minute ending at t (or the next minute with data within the
    hour); None without data."""
    coin_end = CLOCK0 + (int(m["last"][c]) + 1) * HOUR
    probe = next(minutes_from(symbol, t - MINUTE, min(t + HOUR, coin_end)), None)
    if probe is None:
        return None
    bars = minutes_from(symbol, probe[0] + MINUTE, coin_end)
    times, rates, _ = funding_prints(symbol, int(m["first"][c]), int(m["last"][c]))
    i = bisect.bisect_right(times, probe[0])
    prints = ((times[k], rates[k]) for k in range(i, len(times)))
    result = deal(bars, probe[0], probe[3], equity, deep, costs, prints, until)
    result.update(coin=symbol, start=probe[0] + MINUTE, before=equity, entry=probe[3])
    return result


def run(selection, deep, start, end, costs):
    """The rule from start to end, one deal at a time; returns the deals."""
    m = market()
    names, candidates = m["names"], m[selection]
    equity, t, deals = 1.0, start, []
    while t < end:
        h = (t - CLOCK0) // HOUR - 1
        result = None
        for c in candidates[h]:
            if c < 0:
                break
            result = open_deal(str(names[c]), c, m, t, equity, deep, costs, end)
            if result:
                break
        if result is None:
            t = (t // HOUR + 1) * HOUR
            continue
        deals.append(result)
        equity = result["equity"]
        if result["exit"] in ("ruin", "open"):
            break
        t = result["end"] + RESTART
    return deals


def prefetch(pairs):
    """Download (symbol, month) minute files and those coins' funding in parallel, before a sequential run needs
    them."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        list(pool.map(lambda pair: binance_archive.minutes(*pair), sorted(pairs)))
        list(pool.map(binance_archive.funding, sorted({symbol for symbol, _ in pairs})))


def candidate_months(selection, start, end):
    """The months each rule's first pick would need, and the month after (deals run on)."""
    m = market()
    pairs = set()
    for h in range((start - CLOCK0) // HOUR - 1, (end - CLOCK0) // HOUR):
        month = month_of(CLOCK0 + h * HOUR)
        for c in m[selection][h, :1]:
            if c >= 0:
                pairs |= {(str(m["names"][c]), month), (str(m["names"][c]), next_month(month))}
    last = month_of(end - 1)
    return {pair for pair in pairs if pair[1] <= last}


def daily(deals, start, end):
    """Equity at each UTC midnight from start to end, carried between deals, and the daily returns."""
    events = sorted(e for d in deals for e in [(d["start"], d["before"])] + d["marks"] + [(d["end"], d["equity"])])
    times = [t for t, _ in events]
    days = np.arange(start, end + 1, DAY)
    equity = np.array([events[i][1] if i >= 0 else 1.0 for i in (bisect.bisect_right(times, day) - 1 for day in days)])
    equity = np.maximum(equity, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.nan_to_num(equity[1:] / equity[:-1] - 1)
    return days, equity, returns


def references(days):
    """Daily returns of B (long BTC) and A (the 10 coins with most 24h volume, equal weight) between midnights."""
    m = market()
    names = list(m["names"])
    index = (days - CLOCK0) // DAY
    btc = m["midnight"][index, names.index("BTCUSDT")]
    pool_returns = []
    for i in index[:-1]:
        pool = m["volatile"][24 * i - 1]
        pool = pool[pool >= 0]
        pool_returns.append(np.mean(m["midnight"][i + 1, pool] / m["midnight"][i, pool] - 1) if len(pool) else 0.0)
    return btc[1:] / btc[:-1] - 1, np.array(pool_returns)


def summary(deals, start, end):
    days, equity, returns = daily(deals, start, end)
    years = (end - start) / DAY / 365
    return {"days": days, "equity": equity, "returns": returns, "total": equity[-1] - 1,
            "cagr": equity[-1] ** (1 / years) - 1, "vol": returns.std(ddof=1) * np.sqrt(365),
            "sharpe": returns.mean() / returns.std(ddof=1) * np.sqrt(365),
            "drawdown": (equity / np.maximum.accumulate(equity) - 1).min(),
            "ruin": any(d["exit"] == "ruin" for d in deals)}


def equity_at(s, day):
    return s["equity"][(ms(day) - s["days"][0]) // DAY]


def half_years(s, start, end):
    """Equity growth in each calendar half-year from 2020H1 to 2025H2 (the first one from the period's start)."""
    out = {}
    for year in range(2020, 2026):
        for half, (a, b) in enumerate(((f"{year}-01-01", f"{year}-07-01"), (f"{year}-07-01", f"{year + 1}-01-01"))):
            a = max(ms(a), start)
            out[f"{year}H{half + 1}"] = s["equity"][(ms(b) - start) // DAY] / s["equity"][(a - start) // DAY] - 1
    return out


def without_best(s):
    """Growth without the 5 best days, and without the best calendar month."""
    returns = s["returns"]
    months = np.array([month_of(t) for t in s["days"][:-1]])
    by_month = {month: np.prod(1 + returns[months == month]) for month in set(months)}
    best = max(by_month, key=by_month.get)
    return np.prod(1 + np.sort(returns)[:-5]) - 1, np.prod(1 + returns[months != best]) - 1, best


def describe_deals(deals, start, end):
    """Activity against the vault's: deals a day, rungs per deal, time in a deal, exits, the worst deals."""
    rungs = collections.Counter(min(d["rungs"], 6) for d in deals)
    exits = collections.Counter(d["exit"] for d in deals)
    busy = sum(d["end"] - d["start"] for d in deals) / (end - start)
    worst = sorted(deals, key=lambda d: d["equity"] / d["before"])[:5]
    lines = [f"  {len(deals) / ((end - start) / DAY):.1f} deals a day (vault 6.0), in a deal {busy:.0%} of the time (vault 87%)",
             "  rungs per deal: " + ", ".join(f"{k if k < 6 else '6 (cap)'}: {rungs[k] / len(deals):.0%}" for k in sorted(rungs)) +
             " (vault: 63% none, 16% one, 9% two, 6% three, 2% past five)",
             "  exits: " + ", ".join(f"{k} {v}" for k, v in exits.most_common()),
             f"  summed over deals as a share of equity: funding received {-sum(d['funding'] / d['before'] for d in deals):+.1%}, "
             f"fees {sum(d['fees'] / d['before'] for d in deals):.1%}",
             "  worst deals: " + "; ".join(
                 f"{d['coin']} {dt.date.fromtimestamp(d['start'] / 1000)} {d['rungs']} rungs, {d['exit']}, "
                 f"{d['equity'] / d['before'] - 1:+.1%}" for d in worst)]
    return lines


def crash_weeks(deals, s):
    lines = []
    for name, (a, b) in CRASHES.items():
        inside = [d for d in deals if d["start"] < ms(b) and d["end"] > ms(a)]
        held = collections.defaultdict(list)
        for d in inside:
            held[d["coin"]].append(d)
        lines.append(f"  {name}: {equity_at(s, b) / equity_at(s, a) - 1:+.1%}; " + "; ".join(
            f"{coin} {len(ds)} deals, worst {min(d['equity'] / d['before'] - 1 for d in ds):+.1%} ({max(d['rungs'] for d in ds)} rungs)"
            for coin, ds in sorted(held.items(), key=lambda kv: min(d["equity"] / d["before"] for d in kv[1]))[:4]))
    return lines


def report(runs, stress, start, end):
    centre = summary(runs["centre"], start, end)
    b, a = references(centre["days"])
    alpha, exposures = stats.alpha(centre["returns"], [b, a])
    deflated = {n: stats.deflated_sharpe(alpha, n) for n in (TRIALS, 100, 5)}
    halves = half_years(centre, start, end)
    no_days, no_month, best_month = without_best(centre)
    stressed = summary(stress, start, end)
    totals = {name: summary(deals, start, end) for name, deals in runs.items()}
    gates = {"G1": deflated[TRIALS]["dsr"] >= 0.80, "G2": stressed["total"] > 0,
             "G3": np.mean([g > 0 for g in halves.values()]) >= 0.55,
             "G5": centre["drawdown"] >= -2.5 * centre["vol"], "G7": no_days > 0 and no_month > 0,
             "G8": sum(t["total"] > 0 for t in totals.values()) >= 4, "no ruin": not centre["ruin"]}
    lines = [f"== holdout {dt.date.fromtimestamp(start / 1000)} to {dt.date.fromtimestamp(end / 1000)}, centre rule"]
    lines.append(f"  {centre['total']:+.1%} ({centre['cagr']:+.1%} a year), Sharpe {centre['sharpe']:.2f}, vol {centre['vol']:.1%}, "
                 f"worst drawdown {centre['drawdown']:.1%}, worst inside a deal {min(d['lowest'] / d['before'] for d in runs['centre']) - 1:+.1%}")
    lines.append(f"  G1 alpha over B (exposure {exposures[0]:+.2f}) and A ({exposures[1]:+.2f}): Sharpe {deflated[TRIALS]['sharpe']:.2f} over "
                 f"{deflated[TRIALS]['effective']:,.0f} effective days; DSR {deflated[TRIALS]['dsr']:.2f} at N {TRIALS:,} "
                 f"(luck bar {deflated[TRIALS]['luck']:.2f}), {deflated[100]['dsr']:.2f} at 100, {deflated[5]['dsr']:.2f} at 5")
    lines.append(f"  G2 at 10 bps a fill: {stressed['total']:+.1%}")
    lines.append(f"  G3 half-years profitable {sum(g > 0 for g in halves.values())} of {len(halves)}: " +
                 ", ".join(f"{k} {v:+.0%}" for k, v in halves.items()))
    lines.append(f"  G5 worst drawdown {centre['drawdown']:.1%} against 2.5 x vol = {-2.5 * centre['vol']:.1%}")
    lines.append(f"  G7 without the 5 best days {no_days:+.1%}, without the best month ({best_month}) {no_month:+.1%}")
    lines.append("  G8 family at base costs: " + ", ".join(
        f"{name} {t['total']:+.1%} (DD {t['drawdown']:.0%}{', ruin' if t['ruin'] else ''})" for name, t in totals.items()))
    lines.append("  gates: " + ", ".join(f"{k} {'pass' if v else 'FAIL'}" for k, v in gates.items()))
    lines.append("  by year: " + ", ".join(
        f"{y} {equity_at(centre, f'{y + 1}-01-01' if y < 2026 else '2026-02-01') / equity_at(centre, max(f'{y}-01-01', '2020-01-08')) - 1:+.0%}"
        for y in range(2020, 2026)))
    lines += describe_deals(runs["centre"], start, end)
    lines.append("  crash weeks:")
    lines += crash_weeks(runs["centre"], centre)
    for name in ("hold", "stop"):
        lines.append(f"  crash weeks, {name}:")
        lines += crash_weeks(runs[name], totals[name])
    m, held = market(), {d["coin"] for d in runs["centre"]}
    made_up = sum(funding_prints(str(name), int(first), int(last))[2]
                  for name, first, last in zip(m["names"], m["first"], m["last"]) if name in held)
    lines.append(f"  made-up funding prints on coins the centre held: {made_up}")
    return "\n".join(lines)


def check():
    """The vault's deal starts replayed on Binance minute paths, against what the vault did."""
    from decode_grid import account_values, long_fills, trades
    m = market()
    names = [str(n) for n in m["names"]]
    account_value = account_values(json.load(open("data/cold_process/details.json")))
    done, _ = trades(long_fills("cold_process"))
    done = [t for t in done if t["start"] < SEEN[1]]
    prefetch({(binance_symbol(t["coin"]), month) for t in done if binance_symbol(t["coin"]) in names
              for month in (month_of(t["start"]), next_month(month_of(t["start"]))) if month <= month_of(SEEN[1] - 1)})
    pairs, missing, apart = [], collections.Counter(), 0
    for t in done:
        symbol = binance_symbol(t["coin"])
        if symbol not in names:
            missing[t["coin"]] += 1
            continue
        ntl, sz = next(iter(t["buys"].values()))
        result = open_deal(symbol, names.index(symbol), m, t["start"] // MINUTE * MINUTE + MINUTE, 1.0, "unwind", BASE, SEEN[1])
        if result is None or abs(result["entry"] / (ntl / sz) - 1) > 0.02:
            apart += 1
            continue
        pairs.append((t, result))
    pickle.dump([r for _, r in pairs], open("data/cold_process/grid_check_deals.pkl", "wb"))
    vault = [(len(t["buys"]) - 1, (t["end"] - t["start"]) / HOUR, t["max_ntl"] / account_value(t["start"])) for t, _ in pairs]
    engine = [(r["rungs"], (r["end"] - r["start"]) / HOUR, r["peak"]) for _, r in pairs]
    print(f"== engine check: {len(pairs)} of {len(done)} vault deals replayed; no Binance perp {sum(missing.values())} "
          f"({', '.join(f'{c} {k}' for c, k in missing.most_common(6))}), entry more than 2% apart or no minutes {apart}")
    for label, rows in (("vault ", vault), ("engine", engine)):
        rungs = collections.Counter(min(r, 6) for r, _, _ in rows)
        print(f"  {label} rungs: " + ", ".join(f"{k}: {rungs[k] / len(rows):.0%}" for k in range(7)) +
              f"; median {np.median([h for _, h, _ in rows]):.1f} h, p90 {np.percentile([h for _, h, _ in rows], 90):.1f} h; "
              f"peak at 1x equity or more {np.mean([p >= 1 for _, _, p in rows]):.1%}")
    same = np.mean([min(v[0], 6) == min(e[0], 6) for v, e in zip(vault, engine)])
    print(f"  same rung count deal by deal {same:.0%}; engine exits: " +
          ", ".join(f"{k} {v}" for k, v in collections.Counter(r["exit"] for _, r in pairs).most_common()))
    vault_pnl = sum(sum(p for _, _, p in t["sells"]) / account_value(t["start"]) for t, _ in pairs)
    print(f"  P&L summed over these deals, as a share of equity at each entry: vault {vault_pnl:+.1%} (realized, before fees), "
          f"engine {sum(r['equity'] - 1 for _, r in pairs):+.1%}")


def fit():
    """Seen period: how often each selection rule's first pick, or one of its first three, is the coin the vault
    entered at that moment."""
    from decode_grid import long_fills, trades
    m = market()
    names = [str(n) for n in m["names"]]
    done, _ = trades(long_fills("cold_process"))
    starts = [(t["start"], binance_symbol(t["coin"])) for t in done if SEEN[0] <= t["start"] < SEEN[1]]
    print(f"== selection against the vault's {len(starts)} deal starts, {month_of(SEEN[0])} to {month_of(SEEN[1] - 1)}; "
          f"HL's list is the universe from {dt.date.fromtimestamp((CLOCK0 + int(m['switch']) * HOUR) / 1000)}")
    for rule in SELECTIONS:
        orders = [[names[c] for c in m[rule][(t - CLOCK0) // HOUR - 1] if c >= 0] for t, _ in starts]
        first = np.mean([order[:1] == [symbol] for order, (_, symbol) in zip(orders, starts)])
        three = np.mean([symbol in order[:3] for order, (_, symbol) in zip(orders, starts)])
        print(f"  {rule}: first pick {first:.0%}, in the first three {three:.0%}")


def seen():
    prefetch(candidate_months("volatile", *SEEN))
    deals = run("volatile", "unwind", *SEEN, BASE)
    pickle.dump(deals, open("data/cold_process/grid_seen_deals.pkl", "wb"))
    s = summary(deals, *SEEN)
    print(f"== the centre rule on the vault's months, {dt.date.fromtimestamp(SEEN[0] / 1000)} to {dt.date.fromtimestamp(SEEN[1] / 1000)}")
    print(f"  {s['total']:+.1%}, worst drawdown {s['drawdown']:.1%}, Sharpe {s['sharpe']:.2f} (vault: +193% to 11-23, worst drawdown -13.0%)")
    print("\n".join(describe_deals(deals, *SEEN)))


def holdout():
    """The one read. The deals are saved before the report, so a bug in the report doesn't cost a second run."""
    out, saved = Path("data/cold_process/grid_holdout.out"), Path("data/cold_process/grid_holdout_deals.pkl")
    if out.exists():
        raise SystemExit(f"the holdout was read once already; its result is in {out}")
    if not saved.exists():
        for selection in SELECTIONS:
            prefetch(candidate_months(selection, *HOLDOUT))
        runs = {name: run(selection, deep, *HOLDOUT, BASE) for name, (selection, deep) in FAMILY.items()}
        pickle.dump((runs, run("volatile", "unwind", *HOLDOUT, STRESS)), open(saved, "wb"))
    runs, stress = pickle.load(open(saved, "rb"))
    text = report(runs, stress, *HOLDOUT)
    out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    {"check": check, "fit": fit, "seen": seen, "holdout": holdout}[sys.argv[1]]()
