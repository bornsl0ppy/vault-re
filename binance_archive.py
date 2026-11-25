"""Binance's public data archive (data.binance.vision) for USDT-margined perps, delisted coins included: which
coins existed, their hourly and minute klines and their funding prints, cached as npz under data/binance_um/.
Read-only, no keys.

    python binance_archive.py hourly        # every coin's 1h klines -> data/binance_um/1h/SYMBOL.npz
"""
import concurrent.futures
import csv
import io
import json
import re
import sys
import threading
import time
import urllib.parse
import zipfile
from pathlib import Path

import numpy as np
import requests

BUCKET = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
FILES = "https://data.binance.vision/"
EXCHANGE_INFO = "https://fapi.binance.com/fapi/v1/exchangeInfo"
ROOT = Path("data/binance_um")
# Not in today's exchangeInfo (delisted) or listed there as coins, but not coins we'd trade
NOT_COINS = {"BLUEBIRDUSDT", "FOOTBALLUSDT", "DEFIUSDT", "BTCDOMUSDT", "ALLUSDT", "USDCUSDT", "BUSDUSDT", "TUSDUSDT",
             "USDPUSDT", "FDUSDUSDT", "DAIUSDT", "USD1USDT", "PAXGUSDT", "XAUTUSDT"}


local = threading.local()


def get(url, tries=6):
    """The body at url, None if there's no such file. One kept-alive session per thread: dozens of fresh
    connections at once start failing with "no route to host"."""
    if not hasattr(local, "session"):
        local.session = requests.Session()
    for attempt in range(tries):
        try:
            response = local.session.get(url, timeout=60)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == tries - 1:
                raise
            time.sleep(2 ** attempt)


def listing(prefix, tag):
    """S3 listing under prefix: tag "Key" for files, "Prefix" for folders."""
    out, marker = [], ""
    while True:
        body = get(f"{BUCKET}?delimiter=/&prefix={urllib.parse.quote(prefix)}&marker={urllib.parse.quote(marker)}").decode()
        found = [urllib.parse.unquote(x) for x in re.findall(rf"<{tag}>([^<]+)</{tag}>", body)]
        out += [x for x in found if x != prefix]
        if "<IsTruncated>true</IsTruncated>" not in body:
            return out
        marker = re.findall(r"<Key>([^<]+)</Key>|<Prefix>([^<]+)</Prefix>", body)[-1]
        marker = urllib.parse.unquote(marker[0] or marker[1])


def coins():
    """USDT perps in the archive that are coins: no stocks, metals, indices or stablecoins. Today's exchangeInfo is
    saved beside it, since it decides the exclusions."""
    path = ROOT / "coins.json"
    if path.exists():
        return json.load(open(path))["coins"]
    ROOT.mkdir(parents=True, exist_ok=True)
    info = json.loads(get(EXCHANGE_INFO))
    json.dump(info, open(ROOT / "exchange_info.json", "w"))
    listed = {s["symbol"]: s for s in info["symbols"]}
    names = [p.rstrip("/").split("/")[-1] for p in listing("data/futures/um/monthly/klines/", "Prefix")]
    kept, excluded = [], {}
    for name in names:
        if not name.endswith("USDT") or "_" in name:
            continue
        spec = listed.get(name)
        if name in NOT_COINS:
            excluded[name] = "not a coin"
        elif spec and (spec["contractType"] != "PERPETUAL" or spec["underlyingType"] != "COIN"):
            excluded[name] = f"{spec['contractType']} {spec['underlyingType']}"
        else:
            kept.append(name)
    json.dump({"coins": kept, "excluded": excluded}, open(path, "w"), indent=1, ensure_ascii=False)
    return kept


def rows(blob):
    """CSV rows of a zipped archive file, header skipped (older files have none)."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        text = z.read(z.namelist()[0]).decode()
    out = list(csv.reader(io.StringIO(text)))
    return out[1:] if out and not out[0][0].lstrip("-").isdigit() else out


def milliseconds(t):
    return t // 1000 if t > 10**14 else t


def klines(blobs, columns):
    """Kline rows from zips, sorted by open time, duplicates dropped: open time (ms) plus the given columns."""
    data = sorted({milliseconds(int(r[0])): r for blob in blobs for r in rows(blob)}.items())
    out = {"open_time": np.array([t for t, _ in data], dtype=np.int64)}
    for name, index in columns.items():
        out[name] = np.array([float(r[index]) for _, r in data])
    return out


def fetch_all(keys):
    return [blob for blob in (get(FILES + urllib.parse.quote(k)) for k in keys) if blob]


def hourly(symbol):
    """Hourly close and quote volume over the coin's whole life."""
    path = ROOT / "1h" / f"{symbol}.npz"
    if not path.exists():
        keys = [k for k in listing(f"data/futures/um/monthly/klines/{symbol}/1h/", "Key") if k.endswith(".zip")]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **klines(fetch_all(keys), {"close": 4, "quote_volume": 7}))
    return dict(np.load(path))


def minutes(symbol, month):
    """One month (YYYY-MM) of minute high, low and close; daily files when the monthly one isn't out yet."""
    path = ROOT / "1m" / symbol / f"{month}.npz"
    if not path.exists():
        monthly = get(f"{FILES}data/futures/um/monthly/klines/{urllib.parse.quote(symbol)}/1m/{urllib.parse.quote(symbol)}-1m-{month}.zip")
        if monthly:
            blobs = [monthly]
        else:
            keys = listing(f"data/futures/um/daily/klines/{symbol}/1m/{symbol}-1m-{month}-", "Key")
            blobs = fetch_all([k for k in keys if k.endswith(".zip")])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **klines(blobs, {"high": 2, "low": 3, "close": 4}))
    return dict(np.load(path))


def funding(symbol):
    """Funding prints over the coin's whole life: time (ms) and rate."""
    path = ROOT / "funding" / f"{symbol}.npz"
    if not path.exists():
        keys = [k for k in listing(f"data/futures/um/monthly/fundingRate/{symbol}/", "Key") if k.endswith(".zip")]
        prints = sorted({milliseconds(int(r[0])): float(r[2]) for blob in fetch_all(keys) for r in rows(blob)}.items())
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, time=np.array([t for t, _ in prints], dtype=np.int64), rate=np.array([r for _, r in prints]))
    return dict(np.load(path))


if __name__ == "__main__" and sys.argv[1:] == ["hourly"]:
    names = coins()
    with concurrent.futures.ThreadPoolExecutor(16) as pool:
        bars = dict(zip(names, pool.map(lambda s: len(hourly(s)["open_time"]), names)))
    print(f"{len(names)} symbols, {sum(bars.values()):,} hourly bars; empty: {[s for s, n in bars.items() if n == 0]}")
