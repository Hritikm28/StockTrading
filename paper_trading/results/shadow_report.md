# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-27 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.46% per hold after costs, but t=0.22 is too weak to rule out luck.

- Signal days graded: 81 (2026-07-01 to 2026-10-26), prices through 2026-10-26, 159 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -7.82% | -10.24% |
| NIFTY, same timing | -5.52% | -9.95% |
| NIFTY buy and hold | -5.22% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-2.3%**

## Per-cohort (20-day holds)

- Matured cohorts: 58 (about 2 independent)
- Average excess vs NIFTY after costs: 0.46%
- Cohorts that beat NIFTY: 55.2%
- t-stat (overlap-adjusted): 0.22
- Top-5 minus bottom-5: 3.6% per hold, positive 72.4% of days (t=1.11)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0204 | 0.0726 | 0.62 | 58 | 92.4% |
| composite | 0.0455 | 0.0704 | 0.86 | 58 | — |
| mean_rev | 0.0424 | 0.036 | 0.37 | 58 | 48.6% |
| hi52 | -0.0226 | 0.002 | 0.02 | 58 | 72.2% |
| delivery_pct | 0.0291 | -0.0039 | -0.08 | 58 | 93.6% |
| sector_mom | -0.0303 | -0.101 | -1.05 | 58 | 44.5% |
| bulk_deal | — | — | — | 0 | 0.6% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 93.8% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
