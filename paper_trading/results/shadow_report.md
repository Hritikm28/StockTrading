# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-28 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.51% per hold after costs, but t=0.25 is too weak to rule out luck.

- Signal days graded: 82 (2026-07-01 to 2026-10-27), prices through 2026-10-27, 159 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -5.05% | -10.24% |
| NIFTY, same timing | -5.53% | -9.95% |
| NIFTY buy and hold | -5.23% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **0.48%**

## Per-cohort (20-day holds)

- Matured cohorts: 59 (about 2 independent)
- Average excess vs NIFTY after costs: 0.51%
- Cohorts that beat NIFTY: 55.9%
- t-stat (overlap-adjusted): 0.25
- Top-5 minus bottom-5: 3.66% per hold, positive 72.9% of days (t=1.15)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0187 | 0.0699 | 0.61 | 59 | 92.4% |
| composite | 0.0429 | 0.0687 | 0.85 | 59 | — |
| mean_rev | 0.0344 | 0.0338 | 0.35 | 59 | 48.6% |
| hi52 | -0.0176 | -0.0003 | -0.0 | 59 | 72.0% |
| delivery_pct | 0.0306 | -0.0024 | -0.05 | 59 | 93.6% |
| sector_mom | -0.0326 | -0.1043 | -1.09 | 59 | 44.4% |
| bulk_deal | — | — | — | 0 | 0.6% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 93.8% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
