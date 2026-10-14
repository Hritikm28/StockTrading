# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-14 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.83% per hold after costs, but t=0.38 is too weak to rule out luck.

- Signal days graded: 72 (2026-07-01 to 2026-10-13), prices through 2026-10-13, 158 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -2.69% | -5.04% |
| NIFTY, same timing | -5.63% | -9.95% |
| NIFTY buy and hold | -5.35% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **2.94%**

## Per-cohort (20-day holds)

- Matured cohorts: 50 (about 2 independent)
- Average excess vs NIFTY after costs: 0.83%
- Cohorts that beat NIFTY: 60.0%
- t-stat (overlap-adjusted): 0.38
- Top-5 minus bottom-5: 3.87% per hold, positive 72.0% of days (t=1.06)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0473 | 0.1106 | 0.95 | 50 | 93.0% |
| composite | 0.0672 | 0.0883 | 1.01 | 50 | — |
| hi52 | 0.0015 | 0.0344 | 0.39 | 50 | 74.3% |
| mean_rev | 0.0455 | 0.0256 | 0.26 | 50 | 47.8% |
| delivery_pct | 0.0239 | -0.0118 | -0.25 | 50 | 94.3% |
| sector_mom | -0.0216 | -0.1077 | -1.01 | 50 | 45.8% |
| bulk_deal | — | — | — | 0 | 0.7% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 94.3% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
