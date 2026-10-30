# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-30 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.40% per hold after costs, but t=0.2 is too weak to rule out luck.

- Signal days graded: 84 (2026-07-01 to 2026-10-29), prices through 2026-10-29, 159 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.94% | -10.24% |
| NIFTY, same timing | -4.77% | -9.95% |
| NIFTY buy and hold | -4.45% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **0.83%**

## Per-cohort (20-day holds)

- Matured cohorts: 61 (about 3 independent)
- Average excess vs NIFTY after costs: 0.4%
- Cohorts that beat NIFTY: 54.1%
- t-stat (overlap-adjusted): 0.2
- Top-5 minus bottom-5: 3.55% per hold, positive 72.1% of days (t=1.12)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0259 | 0.0668 | 0.6 | 61 | 92.4% |
| composite | 0.0457 | 0.0653 | 0.83 | 61 | — |
| mean_rev | 0.0343 | 0.036 | 0.38 | 61 | 48.5% |
| delivery_pct | 0.0324 | 0.0005 | 0.01 | 61 | 93.6% |
| hi52 | -0.0086 | -0.0026 | -0.03 | 61 | 71.7% |
| sector_mom | -0.0393 | -0.1082 | -1.16 | 61 | 44.0% |
| bulk_deal | — | — | — | 0 | 0.6% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 93.8% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
