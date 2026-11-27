# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-27 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.33% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 104 (2026-07-01 to 2026-11-26), prices through 2026-11-27, 241 stocks priced
- Days the live trend gate allowed trading: 15

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -1.13% | -10.26% |
| NIFTY, same timing | 5.61% | -9.95% |
| NIFTY buy and hold | 6.67% | |
| What the gated live system did | -0.02% | |

Excess of picks over NIFTY (same timing): **-6.74%**

## Per-cohort (20-day holds)

- Matured cohorts: 83 (about 4 independent)
- Average excess vs NIFTY after costs: -1.33%
- Cohorts that beat NIFTY: 43.4%
- t-stat (overlap-adjusted): -0.6
- Top-15 minus bottom-15: 1.6% per hold, positive 66.7% of days (t=0.85)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0139 | 0.0467 | 0.53 | 81 | 61.2% |
| mean_rev | 0.0313 | 0.0406 | 0.54 | 81 | 32.8% |
| composite | 0.0263 | 0.0398 | 0.61 | 81 | — |
| delivery_pct | 0.044 | 0.0344 | 0.66 | 81 | 62.0% |
| hi52 | 0.0111 | -0.0026 | -0.04 | 81 | 47.3% |
| sector_mom | -0.0626 | -0.1212 | -1.47 | 81 | 29.1% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.3% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | -0.0192 | — | — | 0 | 10.2% |
| pead | 0.0017 | — | — | 0 | 7.4% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.0295 | — | — | 0 | 7.9% |
| resid_mom | -0.0905 | — | — | 0 | 8.1% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
