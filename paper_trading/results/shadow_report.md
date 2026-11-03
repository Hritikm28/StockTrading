# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-03 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.63% per hold after costs, but t=0.31 is too weak to rule out luck.

- Signal days graded: 86 (2026-07-01 to 2026-11-02), prices through 2026-11-02, 159 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -5.22% | -10.24% |
| NIFTY, same timing | -2.83% | -9.95% |
| NIFTY buy and hold | -2.35% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-2.39%**

## Per-cohort (20-day holds)

- Matured cohorts: 64 (about 3 independent)
- Average excess vs NIFTY after costs: 0.63%
- Cohorts that beat NIFTY: 56.2%
- t-stat (overlap-adjusted): 0.31
- Top-5 minus bottom-5: 3.98% per hold, positive 73.4% of days (t=1.22)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0484 | 0.0611 | 0.8 | 64 | — |
| momentum | 0.0309 | 0.0609 | 0.57 | 64 | 92.4% |
| mean_rev | 0.0331 | 0.0351 | 0.39 | 64 | 48.4% |
| delivery_pct | 0.0338 | 0.0127 | 0.23 | 64 | 93.5% |
| hi52 | -0.002 | -0.0056 | -0.07 | 64 | 71.4% |
| sector_mom | -0.0406 | -0.1062 | -1.19 | 64 | 43.7% |
| bulk_deal | — | — | — | 0 | 0.6% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 93.8% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
