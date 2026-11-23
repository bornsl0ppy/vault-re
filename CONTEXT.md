# Glossary

**Vault**: a Hyperliquid account a leader trades with followers' deposits; the leader keeps 10% of profits.

**Fill**: one execution against one counterparty. One **order** (oid) can fill many times in one millisecond.

**Chaining**: ordering a millisecond's fills by `startPosition`. A **break** means a missing fill.

**Flow-neutral return**: hourly P&L over the account value going in, so deposits and withdrawals don't count as
performance.

**Episode**: from flat or a direction flip to the next flat or flip.

**Fingerprint**: when a vault enters, whether it adds to winners or losers, and its win/loss shape. Read before
fitting any rule.

**Nightly decision**: a once-a-day vault's order (+convexity: placed at 01:45 UTC).

**Target exposure**: position x price / account value that a rule asks for at a decision.

**Partial adjustment**: moving only part of the way to the target at each decision.

**Persistence baseline**: yesterday's exposure as the prediction. A rule has to beat it.

**Aimed size**: the target a vault was heading for before its partial step, (exposure - (1 - step) x exposure
going in) / step.

**Participation**: 24h volume over the 30-day mean, Astraeus's volume term.

**Replay**: exposure x the next night's BTC return, no costs. **One-step** starts each night from the vault's
own exposure; **free-running** starts from the rule's own, as we'd trade it.

**Amplification**: exposure beyond the rule's size in the rule's direction. Astraeus calls it a size tilt.
