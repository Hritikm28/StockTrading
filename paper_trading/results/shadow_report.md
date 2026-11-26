# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-26 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.28% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 103 (2026-07-01 to 2026-11-25), prices through 2026-11-26, 240 stocks priced
- Days the live trend gate allowed trading: 14

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -2.04% | -10.26% |
| NIFTY, same timing | 6.74% | -9.95% |
| NIFTY buy and hold | 7.83% | |
| What the gated live system did | -0.8% | |

Excess of picks over NIFTY (same timing): **-8.78%**

## Per-cohort (20-day holds)

- Matured cohorts: 82 (about 4 independent)
- Average excess vs NIFTY after costs: -1.28%
- Cohorts that beat NIFTY: 43.9%
- t-stat (overlap-adjusted): -0.58
- Top-15 minus bottom-15: 1.61% per hold, positive 66.2% of days (t=0.85)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0152 | 0.0479 | 0.54 | 80 | 61.5% |
| composite | 0.0279 | 0.0415 | 0.63 | 80 | — |
| mean_rev | 0.0292 | 0.0409 | 0.54 | 80 | 33.0% |
| delivery_pct | 0.0446 | 0.0338 | 0.64 | 80 | 62.3% |
| hi52 | 0.0114 | -0.0029 | -0.04 | 80 | 47.5% |
| sector_mom | -0.0596 | -0.1219 | -1.46 | 80 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.6% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | -0.0155 | — | — | 0 | 9.7% |
| pead | 0.0055 | — | — | 0 | 6.9% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.0077 | — | — | 0 | 7.4% |
| resid_mom | -0.0868 | — | — | 0 | 7.6% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
