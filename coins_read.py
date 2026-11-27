"""+convexity's frozen rule on other coins' perps, as a basket, read once. The spec is in FINDINGS.md (section 4).
The read also judged the basket against our own live strategy; that comparison is private, and this copy compares
with holding BTC instead. Read-only.

    python coins_read.py check     # engine check: BTC through this coin path against backtest.py's own market
    python coins_read.py read      # the one read; refuses a rerun
"""
import sys
from pathlib import Path

import numpy as np

import backtest
import binance_archive
import rule
from backtest import compounded, line, periods, run, summary
from rule import DAY, HOUR, participation
from stats import alpha, deflated_sharpe

COINS = ["ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "BCHUSDT", "LTCUSDT", "YFIUSDT", "EOSUSDT", "ADAUSDT", "BNBUSDT"]
GATED = ("2021-01-01", "2025-08-23")
SEEN = ("2025-08-24", "2026-09-30")
FEE, STRESS = 0.0005, 0.0010
MISSING_FUNDING = 1e-4                       # per 8 h, for a month without prints
TRIALS = 275
GUARD = Path("data/coins/read.out")


class CoinMarket(backtest.Market):
    """backtest.Market's nights for one Binance USDT perp from the public archive: perp closes and quote volume on a
    gapless hourly clock, Binance funding. The coin's life ends at its last hour with volume."""

    def __init__(self, symbol):
        bars = binance_archive.hourly(symbol)
        traded = np.flatnonzero(bars["quote_volume"] > 0)
        life = slice(traded[0], traded[-1] + 1)
        times = bars["open_time"][life]
        opens = np.arange(times[0], times[-1] + HOUR, HOUR)
        at = (times - opens[0]) // HOUR
        close = np.full(len(opens), np.nan)
        close[at] = bars["close"][life]
        volume = np.zeros(len(opens))
        volume[at] = bars["quote_volume"][life]
        self.close = close[np.maximum.accumulate(np.where(np.isfinite(close), np.arange(len(close)), 0))]
        self.first_open = opens[0]
        self.missing = len(opens) - len(times)
        self.days = np.arange(-(-opens[0] // DAY) * DAY, opens[-1] - 25 * HOUR + 1, DAY)
        self.signal = (self.days - opens[0]) // HOUR
        self.fill = self.signal + 1
        self.day_return = self.close[self.fill + 24] / self.close[self.fill] - 1
        self.variance = rule.variance(self.close)[self.signal]
        self.volume = {"perp": participation(volume)[self.signal]}
        prints = binance_archive.funding(symbol)
        times, rates = list(prints["time"]), list(prints["rate"])
        months = {np.datetime64(int(t), "ms").astype("datetime64[M]") for t in times}
        made_up = [t for t in range(-(-opens[0] // (8 * HOUR)) * 8 * HOUR, opens[-1] + HOUR, 8 * HOUR)
                   if np.datetime64(t, "ms").astype("datetime64[M]") not in months]
        self.made_up = len(made_up)
        order = np.argsort(times + made_up, kind="stable")
        self.funding = {"binance": self.day_funding(opens, np.array(times + made_up)[order],
                                                    np.array(rates + [MISSING_FUNDING] * len(made_up))[order])}
        month_ago = np.where(self.signal >= 720, self.signal - 720, 0)
        self.momentum = np.where(self.signal >= 720, np.sign(self.close[self.signal] / self.close[month_ago] - 1), np.nan)
        self.trends = {}

    def live(self, first, last):
        """Nights between first and last whose inputs all exist."""
        return self.between(first, last) & np.isfinite(self.x("perp")) & np.isfinite(self.day_return)


def by_day(series):
    """{day: value} dicts per coin -> the days, and their equal-weight average over the coins present each day."""
    days = sorted(set().union(*series))
    return np.array(days), np.array([np.mean([s[d] for s in series if d in s]) for d in days])


def basket(markets, c, first, last, fee, member=(48, 480)):
    """The equal-weight basket of the rule's daily returns, and each coin's own {day: return}."""
    per_coin = {}
    for symbol, market in markets.items():
        mask = market.live(first, last)
        if mask.any():
            returns, _ = run(market, c, mask, "perp", "binance", fee, member)
            per_coin[symbol] = dict(zip(market.days[mask].tolist(), returns))
    days, returns = by_day(list(per_coin.values()))
    return days, returns, per_coin


def holding(markets, days):
    """Holding BTC's perp on `days`, from each 00:00 UTC close to the next."""
    m = markets["BTCUSDT"]
    at = (days - m.first_open) // HOUR - 1                 # the hourly bar closing at 00:00
    close = m.close[np.clip(at, 0, len(m.close) - 1)]
    return close[1:] / close[:-1] - 1


def sharpe(r):
    return r.mean() / r.std() * np.sqrt(365)


def mixed(candidate, incumbent):
    """Sharpe of a 50/50 mix, each side scaled to the incumbent's volatility."""
    return sharpe(0.5 * incumbent + 0.5 * candidate * incumbent.std() / candidate.std())


def markets_and_c():
    btc = backtest.Market()
    return btc, backtest.scales(btc), {s: CoinMarket(s) for s in COINS + ["BTCUSDT"]}


def check():
    """BTC through CoinMarket (perp closes) against backtest.Market (spot closes), perp volume, base costs, 2019-10 to
    2025-08: already-read data, a check that the coin path computes what the original engine does."""
    btc, c, markets = markets_and_c()
    coin = markets["BTCUSDT"]
    first, last = "2019-10-15", GATED[1]
    spot, _ = run(btc, c, btc.between(first, last), "perp", "binance", FEE)
    perp, _ = run(coin, c, coin.live(first, last), "perp", "binance", FEE)
    days_spot, days_perp = btc.days[btc.between(first, last)], coin.days[coin.live(first, last)]
    common, i, j = np.intersect1d(days_spot, days_perp, return_indices=True)
    print(f"== engine check, BTC {first} to {last}, perp volume, {FEE * 1e4:.0f} bps, Binance funding")
    print(line("backtest.Market, spot closes", summary(spot[i])))
    print(line("CoinMarket, perp closes", summary(perp[j])))
    print(f"  {len(common)} common nights, daily correlation {np.corrcoef(spot[i], perp[j])[0, 1]:.3f}; perp bars "
          f"missing {coin.missing}, funding prints made up {coin.made_up}")


def read():
    if GUARD.exists():
        raise SystemExit(f"{GUARD} exists: the read was done. A rerun needs a bug fix logged with the results.")
    GUARD.parent.mkdir(parents=True, exist_ok=True)
    GUARD.write_text("read started\n")
    lines = []

    def say(text):
        print(text)
        lines.append(text)

    btc, c, markets = markets_and_c()
    coins = {s: markets[s] for s in COINS}
    days, base, per_coin = basket(coins, c, *GATED, FEE)
    _, stress, _ = basket(coins, c, *GATED, STRESS)
    btc = holding(markets, np.concatenate([days, [days[-1] + DAY]]))
    s = summary(base)
    say(f"== +convexity's rule on {len(COINS)} coins, decisions {GATED[0]} to {GATED[1]} ({len(days)} nights), perp "
        f"volume, Binance funding, {FEE * 1e4:.0f} bps a side")
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
    family = {member: compounded(basket(coins, c, *GATED, FEE, member)[1]) for member in backtest.FAMILY}
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
    for symbol, returns in per_coin.items():
        r = np.array(list(returns.values()))
        say(line(f"{symbol.removesuffix('USDT')} ({len(r)} nights)", summary(r)) +
            f"  funding made up {coins[symbol].made_up}")
    pair_days, pair, _ = basket({s: coins[s] for s in ("ETHUSDT", "SOLUSDT")}, c, *GATED, FEE)
    say(line("ETH and SOL alone", summary(pair)))
    coins_held = by_day([dict(zip(m.days[m.live(*GATED)].tolist(), m.day_return[m.live(*GATED)])) for m in coins.values()])[1]
    momentum = by_day([dict(zip(m.days[m.live(*GATED)].tolist(), (m.momentum * m.day_return)[m.live(*GATED)]))
                       for m in coins.values()])[1]
    excess, exposures = alpha(base, [coins_held, momentum])
    measured = {n: deflated_sharpe(excess, n) for n in (5, TRIALS)}
    say(f"alpha over holding the coins and their 30-day momentum: exposures {exposures[0]:+.2f} / {exposures[1]:+.2f}, "
        f"Sharpe {measured[TRIALS]['sharpe']:.2f}; DSR " + ", ".join(f"{measured[n]['dsr']:.2f} at N {n}" for n in measured))
    seen_days, seen, _ = basket(coins, c, *SEEN, FEE)
    say(line(f"seen {SEEN[0]} to {SEEN[1]}", summary(seen)))
    failed = [g for g, ok in gates.items() if not ok]
    say("\n== gates G2-G8: " + ("all pass" if not failed else "failed: " + "; ".join(failed)))
    GUARD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    {"check": check, "read": read}[sys.argv[1]]()
