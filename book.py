"""A vault's position, P&L and account value at any instant, replayed from its fills, funding and deposit ledger
and marked to Binance BTC 15m closes. Events strictly before the instant count. Read-only.

    book = Book("convexity")
    pos, pnl, av = book.at(times_ms)
"""
import json

import numpy as np

from pull import ordered, signed


def flow(x):
    d = x["delta"]
    if d["type"] in ("vaultCreate", "vaultDeposit"):
        return float(d["usdc"])
    if d["type"] == "vaultWithdraw":
        return -float(d["requestedUsd"])                  # the leader's commission leaves the vault too
    return 0.0


def booked(p):
    # before 2026-11-13 funding comes as daily buckets stamped at the day's start; book them at its end so the
    # vault's first bucket doesn't land before vaultCreate
    return p["time"] + 86_400_000 if p["delta"]["nSamples"] else p["time"]


def last_before(times, running, t):
    i = np.searchsorted(times, t, side="left") - 1
    return np.where(i >= 0, running[np.maximum(i, 0)], 0.0)


class Book:
    def __init__(self, name):
        fills = ordered(json.load(open(f"data/{name}/fills_ordered.json")))       # stops on missing fills
        funding = sorted(json.load(open(f"data/{name}/funding.json")), key=booked)
        ledger = sorted(json.load(open(f"data/{name}/ledger.json")), key=lambda x: x["time"])
        candles = json.load(open(f"data/{name}/btc_15m_binance.json"))
        size = np.array([signed(f) for f in fills])
        self.fill_t = np.array([f["time"] for f in fills])
        self.pos_after = np.array([float(f["startPosition"]) for f in fills]) + size
        self.cash_after = np.cumsum(-size * np.array([float(f["px"]) for f in fills]) -
                                    np.array([float(f["fee"]) for f in fills]))
        self.funding_t = np.array([booked(p) for p in funding])
        self.funding_after = np.cumsum([float(p["delta"]["usdc"]) for p in funding])
        self.flow_t = np.array([x["time"] for x in ledger])
        self.flows_after = np.cumsum([flow(x) for x in ledger])
        self.created = min(x["time"] for x in ledger if x["delta"]["type"] == "vaultCreate")
        self.close_t = np.array([c["t"] for c in candles]) + 900_000
        self.close = np.array([float(c["c"]) for c in candles])

    def mark(self, t):
        """BTC at the last 15m close at or before t."""
        return self.close[np.searchsorted(self.close_t, t, side="right") - 1]

    def at(self, t):
        """Position, P&L and account value just before t (ms, a number or an array)."""
        pos = last_before(self.fill_t, self.pos_after, t)
        pnl = (last_before(self.fill_t, self.cash_after, t) + last_before(self.funding_t, self.funding_after, t) +
               pos * self.mark(t))
        return pos, pnl, last_before(self.flow_t, self.flows_after, t) + pnl
