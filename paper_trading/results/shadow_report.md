# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-11 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 0.53% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 92 (2026-07-01 to 2026-11-10), prices through 2026-11-11, 238 stocks priced
- Days the live trend gate allowed trading: 3

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -2.6% | -10.45% |
| NIFTY, same timing | 4.18% | -9.95% |
| NIFTY buy and hold | 5.2% | |
| What the gated live system did | 0.01% | |

Excess of picks over NIFTY (same timing): **-6.78%**

## Per-cohort (20-day holds)

- Matured cohorts: 69 (about 3 independent)
- Average excess vs NIFTY after costs: -0.53%
- Cohorts that beat NIFTY: 49.3%
- t-stat (overlap-adjusted): -0.28
- Top-15 minus bottom-15: 2.0% per hold, positive 72.5% of days (t=1.0)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0403 | 0.0525 | 0.72 | 69 | — |
| momentum | 0.0272 | 0.0492 | 0.48 | 69 | 62.1% |
| mean_rev | 0.0319 | 0.0384 | 0.45 | 69 | 33.3% |
| delivery_pct | 0.0374 | 0.0248 | 0.43 | 69 | 62.9% |
| hi52 | -0.0036 | -0.0165 | -0.2 | 69 | 47.9% |
| sector_mom | -0.0539 | -0.1143 | -1.31 | 69 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.1% |
| fo_ban | — | — | — | 0 | 0.2% |
| mom_riskadj | — | — | — | 0 | 3.6% |
| pead | — | — | — | 0 | 1.6% |
| quality | — | — | — | 0 | 1.4% |
| resid_mom | — | — | — | 0 | 1.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
