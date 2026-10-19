# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-19 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.89% per hold after costs, but t=0.42 is too weak to rule out luck.

- Signal days graded: 75 (2026-07-01 to 2026-10-16), prices through 2026-10-16, 158 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -4.45% | -5.04% |
| NIFTY, same timing | -3.98% | -9.95% |
| NIFTY buy and hold | -3.58% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-0.47%**

## Per-cohort (20-day holds)

- Matured cohorts: 53 (about 2 independent)
- Average excess vs NIFTY after costs: 0.89%
- Cohorts that beat NIFTY: 60.4%
- t-stat (overlap-adjusted): 0.42
- Top-5 minus bottom-5: 3.75% per hold, positive 73.6% of days (t=1.08)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0398 | 0.1009 | 0.9 | 53 | 93.0% |
| composite | 0.0609 | 0.0816 | 0.95 | 53 | — |
| hi52 | -0.01 | 0.0256 | 0.3 | 53 | 73.8% |
| mean_rev | 0.0467 | 0.0139 | 0.14 | 53 | 47.8% |
| delivery_pct | 0.0196 | -0.0157 | -0.34 | 53 | 94.2% |
| sector_mom | -0.0249 | -0.0986 | -0.94 | 53 | 45.8% |
| bulk_deal | — | — | — | 0 | 0.7% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 94.3% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
