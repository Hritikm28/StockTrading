# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-13 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 0.73% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 94 (2026-07-01 to 2026-11-12), prices through 2026-11-13, 238 stocks priced
- Days the live trend gate allowed trading: 5

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -1.47% | -10.45% |
| NIFTY, same timing | 3.75% | -9.95% |
| NIFTY buy and hold | 4.74% | |
| What the gated live system did | 0.18% | |

Excess of picks over NIFTY (same timing): **-5.22%**

## Per-cohort (20-day holds)

- Matured cohorts: 71 (about 3 independent)
- Average excess vs NIFTY after costs: -0.73%
- Cohorts that beat NIFTY: 47.9%
- t-stat (overlap-adjusted): -0.38
- Top-15 minus bottom-15: 1.99% per hold, positive 71.8% of days (t=1.0)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0379 | 0.051 | 0.72 | 71 | — |
| momentum | 0.0271 | 0.0477 | 0.48 | 71 | 62.0% |
| mean_rev | 0.0322 | 0.0381 | 0.45 | 71 | 33.4% |
| delivery_pct | 0.034 | 0.0283 | 0.49 | 71 | 62.9% |
| hi52 | -0.0041 | -0.0175 | -0.22 | 71 | 47.9% |
| sector_mom | -0.062 | -0.1178 | -1.37 | 71 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.1% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0192 | — | — | 0 | 4.8% |
| pead | 0.0701 | — | — | 0 | 2.6% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.1149 | — | — | 0 | 2.4% |
| resid_mom | — | — | — | 0 | 2.6% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
