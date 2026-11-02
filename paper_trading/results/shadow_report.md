# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-02 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.44% per hold after costs, but t=0.22 is too weak to rule out luck.

- Signal days graded: 85 (2026-07-01 to 2026-10-30), prices through 2026-10-30, 159 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -5.44% | -10.24% |
| NIFTY, same timing | -4.32% | -9.95% |
| NIFTY buy and hold | -3.97% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-1.12%**

## Per-cohort (20-day holds)

- Matured cohorts: 62 (about 3 independent)
- Average excess vs NIFTY after costs: 0.44%
- Cohorts that beat NIFTY: 54.8%
- t-stat (overlap-adjusted): 0.22
- Top-5 minus bottom-5: 3.61% per hold, positive 72.6% of days (t=1.15)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0296 | 0.0658 | 0.6 | 62 | 92.4% |
| composite | 0.0478 | 0.065 | 0.84 | 62 | — |
| mean_rev | 0.0335 | 0.0365 | 0.39 | 62 | 48.4% |
| delivery_pct | 0.0337 | 0.0055 | 0.1 | 62 | 93.5% |
| hi52 | -0.0039 | -0.0029 | -0.03 | 62 | 71.5% |
| sector_mom | -0.0391 | -0.1067 | -1.16 | 62 | 43.8% |
| bulk_deal | — | — | — | 0 | 0.6% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 93.8% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
