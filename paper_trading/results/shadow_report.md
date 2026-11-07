# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-07 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by -0.40% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 89 (2026-07-01 to 2026-11-05), prices through 2026-11-06, 238 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -4.16% | -10.45% |
| NIFTY, same timing | 3.32% | -9.95% |
| NIFTY buy and hold | 4.27% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-7.48%**

## Per-cohort (20-day holds)

- Matured cohorts: 68 (about 3 independent)
- Average excess vs NIFTY after costs: -0.4%
- Cohorts that beat NIFTY: 50.0%
- t-stat (overlap-adjusted): -0.22
- Top-15 minus bottom-15: 1.99% per hold, positive 72.1% of days (t=0.98)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0415 | 0.0544 | 0.74 | 68 | — |
| momentum | 0.0261 | 0.0509 | 0.49 | 68 | 63.1% |
| mean_rev | 0.0334 | 0.0393 | 0.45 | 68 | 33.5% |
| delivery_pct | 0.0376 | 0.0238 | 0.41 | 68 | 63.9% |
| hi52 | -0.0028 | -0.015 | -0.18 | 68 | 48.7% |
| sector_mom | -0.0491 | -0.1155 | -1.31 | 68 | 29.6% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 64.1% |
| fo_ban | — | — | — | 0 | 0.2% |
| mom_riskadj | — | — | — | 0 | 2.4% |
| pead | — | — | — | 0 | 0.9% |
| quality | — | — | — | 0 | 0.8% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
