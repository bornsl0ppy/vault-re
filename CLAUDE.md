# vault-re

Rebuild a trading vault's rules from its public fills and orders, then test them. Hyperliquid first. Glossary:
CONTEXT.md. Results: FINDINGS.md.

## Hard rules
- Read-only: public APIs only (Hyperliquid info, Binance). Nothing here signs or sends a transaction.
- No secrets in git. `data/` is gitignored and re-downloadable.
- Run anything that loads more than a few MB under a memory cap:
  `systemd-run --user --scope --quiet -p MemoryMax=4G -p MemorySwapMax=0 nice -n 15 uv run -q --no-project --with numpy <script>`
- Fills must chain: `pull.ordered()` stops on a position break. A missing fill shows up only as a cash offset.
- Third-party write-ups about a vault are hypotheses. Test each against our data before using it.
- Backtests are pre-registered: count every trial, read a holdout once, never tweak and re-read a failed rule.

## Running (from the repo root; NAME is a key in pull.VAULTS)
1. `python pull.py fills NAME` and `orders NAME`, plus candles, funding and ledger into `data/NAME/`; market data
   into `data/btc/` (usage in pull.py's docstring)
2. `python equity.py NAME`: hourly equity through `book.Book`, checked against the vault's own history
3. `python nightly.py NAME`: one row per 01:45 decision, sized by the order's intent
4. `python describe.py NAME`, `match.py`, `daily.py`, `sizing.py`: fingerprint, signal match, first nightly fit
5. `python astraeus_test.py NAME`: the pre-registered test of Astraeus's rule
6. `python backtest.py seen`: engine check; `holdout` was read once on 2026-11-20 and refuses a rerun
7. `python -m pytest tests`: engine accounting, and stats.py against its reference implementation
8. Other vaults: `pull.py fills|orders|twaps|details|ledger|vault-funding NAME`, then `pull.py universe` and
   `pull.py candles-all 4h 2024-08-01 data/candles_4h` (every perp's 4h candles), then `python fingerprint.py NAME`
   (what it is), `python decode_grid.py NAME` (a grid bot's rule) and `python selection.py NAME` (how it picks coins)
9. Cold Process's grid on Binance minutes: `python binance_archive.py hourly` (every USDT perp's hourly klines,
   delisted included; minute files and funding come on demand, ~10 GB), then `python grid_backtest.py check`
   (the vault's own deals replayed), `fit` (selection rules against its picks), `seen` (the rule over its months);
   `holdout` was read on 2026-11-24 and refuses a rerun
10. `python charts.py`: the charts in FINDINGS.md and the README, light and dark, from what steps 2-9 saved
