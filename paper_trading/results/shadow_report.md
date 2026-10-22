# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-22 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.77% per hold after costs, but t=0.37 is too weak to rule out luck.

- Signal days graded: 78 (2026-07-01 to 2026-10-21), prices through 2026-10-21, 158 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -7.48% | -7.52% |
| NIFTY, same timing | -5.32% | -9.95% |
| NIFTY buy and hold | -4.98% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-2.16%**

## Per-cohort (20-day holds)

- Matured cohorts: 55 (about 2 independent)
- Average excess vs NIFTY after costs: 0.77%
- Cohorts that beat NIFTY: 58.2%
- t-stat (overlap-adjusted): 0.37
- Top-5 minus bottom-5: 3.76% per hold, positive 74.5% of days (t=1.12)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0294 | 0.0884 | 0.77 | 55 | 93.0% |
| composite | 0.0497 | 0.0767 | 0.91 | 55 | — |
| mean_rev | 0.0488 | 0.0246 | 0.25 | 55 | 47.9% |
| hi52 | -0.0204 | 0.0138 | 0.15 | 55 | 73.3% |
| delivery_pct | 0.0195 | -0.0098 | -0.21 | 55 | 94.2% |
| sector_mom | -0.0294 | -0.1007 | -1.0 | 55 | 45.5% |
| bulk_deal | — | — | — | 0 | 0.7% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 94.3% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
