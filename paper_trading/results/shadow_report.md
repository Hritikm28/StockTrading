# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-26 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.53% per hold after costs, but t=0.25 is too weak to rule out luck.

- Signal days graded: 80 (2026-07-01 to 2026-10-23), prices through 2026-10-23, 159 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -10.2% | -10.24% |
| NIFTY, same timing | -6.14% | -9.95% |
| NIFTY buy and hold | -5.86% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-4.06%**

## Per-cohort (20-day holds)

- Matured cohorts: 57 (about 2 independent)
- Average excess vs NIFTY after costs: 0.53%
- Cohorts that beat NIFTY: 56.1%
- t-stat (overlap-adjusted): 0.25
- Top-5 minus bottom-5: 3.57% per hold, positive 71.9% of days (t=1.09)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0215 | 0.0768 | 0.66 | 57 | 92.4% |
| composite | 0.0446 | 0.0708 | 0.85 | 57 | — |
| mean_rev | 0.0435 | 0.0335 | 0.34 | 57 | 48.3% |
| hi52 | -0.0248 | 0.0057 | 0.06 | 57 | 72.4% |
| delivery_pct | 0.0256 | -0.0054 | -0.11 | 57 | 93.6% |
| sector_mom | -0.0313 | -0.1006 | -1.03 | 57 | 44.8% |
| bulk_deal | — | — | — | 0 | 0.6% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 93.8% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
