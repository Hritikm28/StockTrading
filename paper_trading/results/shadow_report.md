# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-20 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.38% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 99 (2026-07-01 to 2026-11-19), prices through 2026-11-20, 239 stocks priced
- Days the live trend gate allowed trading: 10

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -2.2% | -10.45% |
| NIFTY, same timing | 6.5% | -9.95% |
| NIFTY buy and hold | 7.48% | |
| What the gated live system did | -0.31% | |

Excess of picks over NIFTY (same timing): **-8.7%**

## Per-cohort (20-day holds)

- Matured cohorts: 77 (about 3 independent)
- Average excess vs NIFTY after costs: -1.38%
- Cohorts that beat NIFTY: 44.2%
- t-stat (overlap-adjusted): -0.64
- Top-15 minus bottom-15: 1.68% per hold, positive 67.5% of days (t=0.87)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0235 | 0.0495 | 0.54 | 77 | 61.8% |
| composite | 0.0342 | 0.0439 | 0.66 | 77 | — |
| mean_rev | 0.0301 | 0.0363 | 0.47 | 77 | 33.3% |
| delivery_pct | 0.0403 | 0.0326 | 0.61 | 77 | 62.6% |
| hi52 | 0.0074 | -0.0037 | -0.05 | 77 | 47.7% |
| sector_mom | -0.0603 | -0.1279 | -1.55 | 77 | 29.3% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.8% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0229 | — | — | 0 | 7.7% |
| pead | 0.0443 | — | — | 0 | 5.0% |
| pledge | — | — | — | 0 | 0.0% |
| quality | -0.002 | — | — | 0 | 5.3% |
| resid_mom | -0.0269 | — | — | 0 | 5.5% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
