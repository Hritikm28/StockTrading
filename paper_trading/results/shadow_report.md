# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-06 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by -0.21% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 89 (2026-07-01 to 2026-11-05), prices through 2026-11-05, 237 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.61% | -10.45% |
| NIFTY, same timing | 1.38% | -9.95% |
| NIFTY buy and hold | 2.21% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-4.99%**

## Per-cohort (20-day holds)

- Matured cohorts: 67 (about 3 independent)
- Average excess vs NIFTY after costs: -0.21%
- Cohorts that beat NIFTY: 50.7%
- t-stat (overlap-adjusted): -0.13
- Top-15 minus bottom-15: 2.05% per hold, positive 73.1% of days (t=1.0)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0421 | 0.0557 | 0.75 | 67 | — |
| momentum | 0.0264 | 0.0526 | 0.5 | 67 | 62.4% |
| mean_rev | 0.0328 | 0.0373 | 0.42 | 67 | 32.9% |
| delivery_pct | 0.0356 | 0.021 | 0.36 | 67 | 63.2% |
| hi52 | -0.001 | -0.0133 | -0.16 | 67 | 48.2% |
| sector_mom | -0.0445 | -0.1147 | -1.28 | 67 | 29.4% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.4% |
| fo_ban | — | — | — | 0 | 0.2% |
| mom_riskadj | — | — | — | 0 | 1.7% |
| pead | — | — | — | 0 | 0.5% |
| quality | — | — | — | 0 | 0.6% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
