# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-18 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.24% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 97 (2026-07-01 to 2026-11-17), prices through 2026-11-18, 239 stocks priced
- Days the live trend gate allowed trading: 8

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.27% | -10.45% |
| NIFTY, same timing | 4.1% | -9.95% |
| NIFTY buy and hold | 5.09% | |
| What the gated live system did | -0.9% | |

Excess of picks over NIFTY (same timing): **-7.37%**

## Per-cohort (20-day holds)

- Matured cohorts: 75 (about 3 independent)
- Average excess vs NIFTY after costs: -1.24%
- Cohorts that beat NIFTY: 45.3%
- t-stat (overlap-adjusted): -0.57
- Top-15 minus bottom-15: 1.75% per hold, positive 69.3% of days (t=0.89)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0271 | 0.0477 | 0.51 | 75 | 61.8% |
| composite | 0.0373 | 0.0453 | 0.66 | 75 | — |
| mean_rev | 0.03 | 0.0366 | 0.46 | 75 | 33.3% |
| delivery_pct | 0.0318 | 0.0311 | 0.57 | 75 | 62.6% |
| hi52 | 0.0026 | -0.0093 | -0.12 | 75 | 47.7% |
| sector_mom | -0.0687 | -0.1205 | -1.48 | 75 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.8% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0617 | — | — | 0 | 6.5% |
| pead | 0.0452 | — | — | 0 | 4.0% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.0436 | — | — | 0 | 4.1% |
| resid_mom | 0.032 | — | — | 0 | 4.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
