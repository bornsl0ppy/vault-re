# vault-re

Reverse-engineering Hyperliquid vaults from their public fills and orders, then checking whether the rules hold up as
strategies on their own.

First vault: [+convexity](https://app.hyperliquid.xyz/vaults/0x5661a070eb13c7c55ac3210b2447d4bea426cbf5), a BTC trend
follower that trades once a day. Inspired by [Astraeus's decode](https://x.com/AstraeusCap/status/2122868247972180231).
We rebuilt the vault from scratch, checked his claims, then backtested the rule over 2017-2025 with fees and funding,
pre-registered, holdout read once.

The rule tracks the vault well. Run on its own it's a low-leverage BTC trend follower whose edge over buy-and-hold
and plain momentum is within luck. His cross-sectional sleeve reproduces on our own Binance perp data, but it's
mostly plain momentum and lost in 2023-24. Numbers in [FINDINGS.md](FINDINGS.md).

## Code
- `pull.py`: complete vault fills and orders, Binance and Hyperliquid market data. Public APIs, no keys.
- `book.py`, `equity.py`: position, P&L and account value at any point, checked against the vault's own history.
- `nightly.py`: one row per daily decision, sized by order intent.
- `rule.py`, `astraeus_test.py`: the decoded rule and the test of the article's claims.
- `backtest.py`, `stats.py`: nightly engine with fees, funding and drift; deflated Sharpe.
- `describe.py`, `match.py`, `daily.py`, `sizing.py`: first exploratory passes.

Steps in `CLAUDE.md`. Runs on Python with numpy and scipy (`uv run --no-project --with numpy --with scipy python
<script>`). Data isn't in the repo, `pull.py` fetches it. Read-only, nothing here trades.

MIT licensed.
