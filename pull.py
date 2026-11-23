"""Pull a Hyperliquid vault's full fill and order history and the market data to read it against, from public
APIs (Hyperliquid info, Binance). Read-only, no keys.

    python pull.py fills convexity                       # -> data/convexity/fills_ordered.json
    python pull.py orders convexity                      # -> data/convexity/orders.json
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

INFO = "https://api.hyperliquid.xyz/info"
VAULTS = {"convexity": "0x5661a070eb13c7c55ac3210b2447d4bea426cbf5"}
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
    return ordered(list(out.values()))


def signed(f):
    return float(f["sz"]) * (1 if f["side"] == "B" else -1)


def ordered(fills):
    """Fills in execution order. The API's order within a millisecond is arbitrary, so chain them by
    startPosition; a break means a missing fill, and every P&L built on top would carry a cash offset."""
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
    breaks = [(a["time"], b["time"]) for a, b in zip(out, out[1:])
              if abs(float(b["startPosition"]) - (float(a["startPosition"]) + signed(a))) > 1e-6]
    if breaks:
        raise SystemExit(f"{len(breaks)} position breaks, first at {breaks[0]}: fills are missing")
    return out


def orders(user):
    """Every order with its placement time, origSz and limitPx, one row per status change. The API keeps the
    2,000 most recent rows."""
    rows = info({"type": "historicalOrders", "user": user})
    if len(rows) >= 2000:
        raise SystemExit("historicalOrders hit its 2,000-row cap: the oldest orders are gone")
    return rows


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
    step = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "1d": 86400}[interval] * 1000
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
        data, out = fills(VAULTS[args[0]]), f"data/{args[0]}/fills_ordered.json"
    elif kind == "orders":
        data, out = orders(VAULTS[args[0]]), f"data/{args[0]}/orders.json"
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
    json.dump(data, open(out, "w"))
    print(len(data), "records ->", out, file=sys.stderr)
