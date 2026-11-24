"""Pull a Hyperliquid vault's full fill and order history and the market data to read it against, from public
APIs (Hyperliquid info, Binance). Read-only, no keys.

    python pull.py fills convexity                       # -> data/convexity/fills_ordered.json
    python pull.py orders convexity                      # -> data/convexity/orders.json
    python pull.py ledger convexity                      # deposits and withdrawals -> data/convexity/ledger.json
    python pull.py vault-funding convexity               # funding paid and received -> data/convexity/funding.json
    python pull.py details convexity                     # vaultDetails: leader, followers, history -> data/convexity/details.json
    python pull.py twaps convexity                       # TWAPs, whose slices the fills leave out -> data/convexity/twaps.json
    python pull.py candles BTC 15m 2025-08-01 data/convexity/btc_15m.json            # Hyperliquid
    python pull.py klines spot BTCUSDT 1h 2025-01-01 data/btc/spot_1h.json           # Binance; perp for USDT-M
    python pull.py funding BTCUSDT 2025-01-01 data/btc/funding.json                  # Binance USDT-M prints
    python pull.py hl-funding BTC 2023-05-01 data/btc/hl_funding.json                # Hyperliquid hourly prints
"""
import datetime as dt
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

INFO = "https://api.hyperliquid.xyz/info"
VAULTS = {"convexity": "0x5661a070eb13c7c55ac3210b2447d4bea426cbf5",
          "delta_01": "0x3005fade4c0df5e1cd187d7062da359416f0eb8e",
          "22cap": "0xba939edf38c0ae0cc689c98b492e0535f43e4550",
          "cold_process": "0xc3b1bf1f1e6fb8161ef4dc2f34e2c56f70b94b11",
          "futurai": "0xb65dd7c56afbf3b272ab5fc49be44b47dca18003",
          "pf1": "0xa1b6d8efbcb2fb750a84dbc05649fa4968034f04",
          "citadel": "0xda51323fe9800c8365646ad5c7ade0dd17fdc167"}
KLINES = {"spot": "https://api.binance.com/api/v3/klines", "perp": "https://fapi.binance.com/fapi/v1/klines"}
FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"


def info(body):
    req = urllib.request.Request(INFO, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))


def fills(user):
    """Every fill the API still serves (it keeps the 10,000 most recent), deduplicated by tid. Each page restarts
    at the newest millisecond it reached: one order fills many times in a millisecond, and a 2,000-fill page can
    end inside one (starting at newest + 1 lost 4 fills on +convexity)."""
    out, start = {}, 0
    while True:
        page = info({"type": "userFillsByTime", "user": user, "startTime": start, "endTime": int(time.time() * 1000)})
        time.sleep(1.0)
        new = [f for f in page if f["tid"] not in out]
        for f in page:
            out[f["tid"]] = f
        if not new:
            break
        newest = max(f["time"] for f in page)
        print(len(out), "fills so far, up to", dt.datetime.fromtimestamp(newest / 1000, dt.timezone.utc), file=sys.stderr, flush=True)
        if len(page) < 2000:
            break
        start = newest
    return list(out.values())


def signed(f):
    return float(f["sz"]) * (1 if f["side"] == "B" else -1)


def is_perp(coin):
    return "/" not in coin and not coin.startswith("@")


def ordered(fills):
    """Fills in execution order; stops on a perp position break (a missing fill), which would leave every P&L
    built on top with a cash offset."""
    out, breaks = chained(fills)
    if breaks:
        raise SystemExit(f"{len(breaks)} position breaks, first at {breaks[0]}: fills are missing")
    return out


def chained(fills):
    """Fills in execution order and the perp position breaks (coin, time before, time after). The API's order within
    a millisecond is arbitrary, so each coin's fills are chained by startPosition; spot balances also move with
    transfers, so spot chains aren't checked."""
    by_coin = {}
    for f in fills:
        by_coin.setdefault(f["coin"], []).append(f)
    out, breaks = [], []
    for coin, group in by_coin.items():
        chain = _chained(group)
        out += chain
        if is_perp(coin):
            breaks += [(coin, a["time"], b["time"]) for a, b in zip(chain, chain[1:])
                       if abs(float(b["startPosition"]) - (float(a["startPosition"]) + signed(a)))
                       > 1e-6 * max(1.0, abs(float(b["startPosition"])))]
    return sorted(out, key=lambda f: f["time"]), sorted(breaks, key=lambda b: b[2])   # stable: keeps each chain


def _chained(fills):
    """One coin's fills in time order, each millisecond's fills chained from the position the last one left."""
    by_ms = {}
    for f in fills:
        by_ms.setdefault(f["time"], []).append(f)
    out, pos = [], None
    for t in sorted(by_ms):
        group = by_ms[t]
        while group:
            nxt = next((f for f in group if pos is None or abs(float(f["startPosition"]) - pos) < 1e-6), None)
            if nxt is None:
                nxt = min(group, key=lambda f: abs(float(f["startPosition"]) - pos))
            group.remove(nxt)
            out.append(nxt)
            pos = float(nxt["startPosition"]) + signed(nxt)
    return out


def orders(user):
    """Every order with its placement time, origSz and limitPx, one row per status change. The API keeps the
    2,000 most recent rows."""
    rows = info({"type": "historicalOrders", "user": user})
    if len(rows) >= 2000:
        print("historicalOrders hit its 2,000-row cap: only the latest orders are here", file=sys.stderr)
    return rows


def account(kind, user, cap):
    """A user's userFunding payments or userNonFundingLedgerUpdates, paged forward by time. Each page restarts at
    its newest record's time and dedupes on the whole record, as funding rows share one empty hash."""
    out, start = {}, 0
    while True:
        page = info({"type": kind, "user": user, "startTime": start, "endTime": int(time.time() * 1000)})
        time.sleep(1.0)
        new = [row for row in page if json.dumps(row, sort_keys=True) not in out]
        for row in page:
            out[json.dumps(row, sort_keys=True)] = row
        if not new or len(page) < cap:
            return sorted(out.values(), key=lambda row: row["time"])
        start = max(row["time"] for row in page)


def binance(url, query, key):
    """Page a Binance list endpoint forward from query['startTime'] to now."""
    out = {}
    while True:
        page = json.load(urllib.request.urlopen(f"{url}?{urllib.parse.urlencode(query | {'limit': 1000})}", timeout=60))
        time.sleep(0.5)
        for row in page:
            out[key(row)] = row
        if len(page) < 1000:
            return [out[t] for t in sorted(out)]
        query["startTime"] = max(out) + 1


def candles(coin, interval, since):
    step = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}[interval] * 1000
    out, start, end = {}, int(since.timestamp() * 1000), int(time.time() * 1000)
    while start < end:
        page = info({"type": "candleSnapshot", "req": {"coin": coin, "interval": interval, "startTime": start,
                                                        "endTime": min(end, start + 5000 * step)}})
        time.sleep(1.0)
        for c in page:
            out[c["t"]] = c
        start += 5000 * step
    return [out[t] for t in sorted(out)]


def hl_funding(coin, since):
    """Hyperliquid's hourly funding prints for a coin from since (ms) to now, 500 per answer."""
    out = {}
    while True:
        page = info({"type": "fundingHistory", "coin": coin, "startTime": since})
        time.sleep(1.0)
        for row in page:
            out[row["time"]] = row
        if len(page) < 500:
            return [out[t] for t in sorted(out)]
        since = max(out) + 1


if __name__ == "__main__":
    kind, args = sys.argv[1], sys.argv[2:]
    def ms(day):
        return int(dt.datetime.fromisoformat(day).replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    if kind == "fills":
        data, breaks = chained(fills(VAULTS[args[0]]))
        out = f"data/{args[0]}/fills_ordered.json"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(breaks, open(f"data/{args[0]}/breaks.json", "w"))
        if breaks:
            last = dt.datetime.fromtimestamp(breaks[-1][2] / 1000, dt.timezone.utc)
            print(f"{len(breaks)} perp position breaks (missing fills), the last at {last:%Y-%m-%d %H:%M}: the record "
                  f"is complete only after it (data/{args[0]}/breaks.json)", file=sys.stderr)
    elif kind == "orders":
        data, out = orders(VAULTS[args[0]]), f"data/{args[0]}/orders.json"
    elif kind == "details":
        data, out = info({"type": "vaultDetails", "vaultAddress": VAULTS[args[0]]}), f"data/{args[0]}/details.json"
    elif kind == "twaps":
        data, out = info({"type": "twapHistory", "user": VAULTS[args[0]]}), f"data/{args[0]}/twaps.json"
    elif kind == "ledger":
        data, out = account("userNonFundingLedgerUpdates", VAULTS[args[0]], 2000), f"data/{args[0]}/ledger.json"
    elif kind == "vault-funding":
        data, out = account("userFunding", VAULTS[args[0]], 500), f"data/{args[0]}/funding.json"
    elif kind == "candles":
        data, out = candles(args[0], args[1], dt.datetime.fromtimestamp(ms(args[2]) / 1000, dt.timezone.utc)), args[3]
    elif kind == "klines":
        market, symbol, interval, since, out = args
        data = binance(KLINES[market], {"symbol": symbol, "interval": interval, "startTime": ms(since)}, lambda r: r[0])
    elif kind == "funding":
        symbol, since, out = args
        data = binance(FUNDING, {"symbol": symbol, "startTime": ms(since)}, lambda r: r["fundingTime"])
    else:
        coin, since, out = args
        data = hl_funding(coin, ms(since))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out, "w"))
    print(len(data), "records ->", out, file=sys.stderr)
