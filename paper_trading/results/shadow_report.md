# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-23 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.65% per hold after costs, but t=0.31 is too weak to rule out luck.

- Signal days graded: 79 (2026-07-01 to 2026-10-22), prices through 2026-10-22, 158 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -8.18% | -8.22% |
| NIFTY, same timing | -5.79% | -9.95% |
| NIFTY buy and hold | -5.49% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-2.39%**

## Per-cohort (20-day holds)

- Matured cohorts: 56 (about 2 independent)
- Average excess vs NIFTY after costs: 0.65%
- Cohorts that beat NIFTY: 57.1%
- t-stat (overlap-adjusted): 0.31
- Top-5 minus bottom-5: 3.65% per hold, positive 73.2% of days (t=1.09)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0252 | 0.0823 | 0.71 | 56 | 93.0% |
| composite | 0.0474 | 0.073 | 0.87 | 56 | — |
| mean_rev | 0.0468 | 0.0289 | 0.29 | 56 | 48.3% |
| hi52 | -0.0239 | 0.0092 | 0.1 | 56 | 73.1% |
| delivery_pct | 0.0229 | -0.0093 | -0.2 | 56 | 94.2% |
| sector_mom | -0.0306 | -0.0988 | -0.99 | 56 | 45.3% |
| bulk_deal | — | — | — | 0 | 0.6% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 94.3% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
