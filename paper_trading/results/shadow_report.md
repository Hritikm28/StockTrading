# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-24 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.56% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 100 (2026-07-01 to 2026-11-20), prices through 2026-11-23, 239 stocks priced
- Days the live trend gate allowed trading: 11

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -4.52% | -10.45% |
| NIFTY, same timing | 4.61% | -9.95% |
| NIFTY buy and hold | 5.6% | |
| What the gated live system did | -1.13% | |

Excess of picks over NIFTY (same timing): **-9.13%**

## Per-cohort (20-day holds)

- Matured cohorts: 78 (about 3 independent)
- Average excess vs NIFTY after costs: -1.56%
- Cohorts that beat NIFTY: 43.6%
- t-stat (overlap-adjusted): -0.68
- Top-15 minus bottom-15: 1.58% per hold, positive 66.7% of days (t=0.8)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0202 | 0.0463 | 0.51 | 78 | 61.8% |
| composite | 0.0322 | 0.0413 | 0.61 | 78 | — |
| delivery_pct | 0.0421 | 0.0357 | 0.66 | 78 | 62.6% |
| mean_rev | 0.0283 | 0.0337 | 0.43 | 78 | 33.3% |
| hi52 | 0.01 | -0.0024 | -0.03 | 78 | 47.7% |
| sector_mom | -0.0579 | -0.1263 | -1.56 | 78 | 29.3% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.8% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0153 | — | — | 0 | 8.2% |
| pead | 0.0333 | — | — | 0 | 5.5% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.0009 | — | — | 0 | 5.8% |
| resid_mom | -0.0334 | — | — | 0 | 6.0% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
