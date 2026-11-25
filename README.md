# vault-re

Reverse-engineering Hyperliquid vaults from their public fills and orders, then checking whether the rules hold up as
strategies on their own.

First vault: [+convexity](https://app.hyperliquid.xyz/vaults/0x5661a070eb13c7c55ac3210b2447d4bea426cbf5), a BTC trend
follower that trades once a day. Inspired by [Astraeus's decode](https://x.com/AstraeusCap/status/2122868247972180231).
We rebuilt the vault from scratch, checked his claims, then backtested the rule over 2017-2025 with fees and funding,
pre-registered, holdout read once.

The rule tracks the vault well. Run on its own it's a low-leverage BTC trend follower whose edge over buy-and-hold
and plain momentum is within luck. His cross-sectional sleeve reproduces on our own Binance perp data, but it's
mostly plain momentum and lost in 2023-24.

![+convexity's rule from 2017 to 2026 against holding BTC: about the same return by the end, with a worst drawdown of -25% against -83%](charts/convexity.png)

Second vault: [Cold Process](https://app.hyperliquid.xyz/vaults/0xc3b1bf1f1e6fb8161ef4dc2f34e2c56f70b94b11), a bot
that buys an alt, buys more as it falls and sells at 1.34% over its average price (a DCA grid). It closed 99.5% of
its deals in profit. How much it buys and when it sells decode cleanly. How it picks coins doesn't, and the picks
carry its record. Run on Binance minute data with a rule picking the coins, it compounds to absurd numbers through
2021, then LUNA wipes it out in May 2022.

![Cold Process's rule from 2020: up 6,308x, then LUNA took it to zero on 2022-05-11; with a stop 10% under its average it survives at +4,465% after falling 97%](charts/cold-process.png)

Numbers in [FINDINGS.md](FINDINGS.md).

## Code
- `pull.py`: complete vault fills and orders, Binance and Hyperliquid market data. Public APIs, no keys.
- `book.py`, `equity.py`: position, P&L and account value at any point, checked against the vault's own history.
- `nightly.py`: one row per daily decision, sized by order intent.
- `rule.py`, `astraeus_test.py`: the decoded rule and the test of the article's claims.
- `backtest.py`, `stats.py`: nightly engine with fees, funding and drift; deflated Sharpe.
- `describe.py`, `match.py`, `daily.py`, `sizing.py`: first exploratory passes.
- `fingerprint.py`: what any vault does, from its fills, TWAPs and positions.
- `decode_grid.py`, `selection.py`: a grid bot's ladder and take-profit, and how it picks its coins.
- `binance_archive.py`: Binance's public archive of USDT perps, delisted coins included.
- `grid_backtest.py`: minute-path engine for the grid, with fees, funding and the pre-registered read.
- `charts.py`: the two charts above.

Steps in `CLAUDE.md`. Runs on Python with numpy, scipy, requests and matplotlib (`uv run --no-project --with numpy
--with scipy --with requests --with matplotlib python <script>`). Data isn't in the repo, `pull.py` and
`binance_archive.py` fetch it. Read-only, nothing here trades.

MIT licensed.
