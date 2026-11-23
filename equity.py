"""Hourly equity of a vault rebuilt from its fills, funding and deposit ledger, marked to BTC closes; checked
against the vault's own reported PnL and account value. Returns are flow-neutral (hourly PnL over the
account value going in), so deposits and withdrawals don't look like performance. Read-only.

    python equity.py convexity
"""
import bisect
import json
import sys

import numpy as np

from book import Book
from pull import VAULTS

name = sys.argv[1]
book = Book(name)
details = json.load(open("data/details.json"))[VAULTS[name]]
h = np.arange((book.created // 3_600_000 + 1) * 3_600_000, book.close_t[-1], 3_600_000)
pos, pnl, av = book.at(h)
mark = book.mark(h)

port = dict(details["portfolio"])["allTime"]
rep_pnl = {t: float(v) for t, v in port["pnlHistory"]}
rep_av = {t: float(v) for t, v in port["accountValueHistory"]}
gaps = []
for t in rep_pnl:
    i = bisect.bisect_right(h, t) - 1
    if i >= 0 and rep_av[t] > 0:                      # the history opens with a 0/0 point four days after creation
        gaps.append((pnl[i] - rep_pnl[t], av[i] - rep_av[t], rep_av[t]))
g = np.array(gaps)
print(f"check vs the vault's own history ({len(g)} points): PnL off by median ${np.median(np.abs(g[:,0])):,.0f} (p90 ${np.percentile(np.abs(g[:,0]),90):,.0f}); "
      f"account value off by median {np.median(np.abs(g[:,1])/g[:,2])*100:.1f}%")

ret = np.diff(pnl) / av[:-1]
ret[~np.isfinite(ret)] = 0
index = np.cumprod(1 + ret)
dd = index / np.maximum.accumulate(index) - 1
lev = np.abs(pos) * mark / av
btc = mark / mark[0]
btc_dd = btc / np.maximum.accumulate(btc) - 1
years = (h[-1] - h[0]) / 3.156e10
print(f"vault (flow-neutral, hourly): total {index[-1]*100-100:+.0f}%, vol {ret.std()*np.sqrt(8760)*100:.0f}%/yr, "
      f"Sharpe {ret.mean()/ret.std()*np.sqrt(8760):.2f}, max drawdown {dd.min()*100:.1f}% (worst at {np.datetime64(int(h[np.argmin(dd)]), 'ms')})")
print(f"BTC held over the same {years:.2f} years: {btc[-1]*100-100:+.0f}%, max drawdown {btc_dd.min()*100:.1f}%")
print(f"leverage |notional| / account value: median {np.median(lev[lev>0]):.2f}x, p90 {np.percentile(lev[lev>0],90):.2f}x, max {lev.max():.2f}x")
print(f"PnL $: total ${pnl[-1]:,.0f}; worst peak-to-trough ${(pnl - np.maximum.accumulate(pnl)).min():,.0f}")
for y in sorted({str(np.datetime64(int(x), 'ms'))[:7] for x in h}):
    m = np.array([str(np.datetime64(int(x), 'ms'))[:7] == y for x in h[1:]])
    if m.any():
        print(f"  {y}: return {np.prod(1 + ret[m])*100-100:+6.1f}%  BTC {mark[1:][m][-1]/mark[:-1][m][0]*100-100:+6.1f}%  median leverage {np.median(lev[1:][m]):.2f}x")
np.savez(f"data/{name}/equity.npz", h=h, pos=pos, mark=mark, pnl=pnl, av=av)
