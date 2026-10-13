# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-13 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.78% per hold after costs, but t=0.34 is too weak to rule out luck.

- Signal days graded: 71 (2026-07-01 to 2026-10-12), prices through 2026-10-12, 158 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.35% | -5.04% |
| NIFTY, same timing | -7.04% | -9.95% |
| NIFTY buy and hold | -6.84% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **3.69%**

## Per-cohort (20-day holds)

- Matured cohorts: 48 (about 2 independent)
- Average excess vs NIFTY after costs: 0.78%
- Cohorts that beat NIFTY: 58.3%
- t-stat (overlap-adjusted): 0.34
- Top-5 minus bottom-5: 4.02% per hold, positive 70.8% of days (t=1.06)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0497 | 0.1177 | 0.99 | 48 | 93.1% |
| composite | 0.0693 | 0.0926 | 1.03 | 48 | — |
| hi52 | 0.0063 | 0.0371 | 0.41 | 48 | 74.4% |
| mean_rev | 0.0489 | 0.0315 | 0.32 | 48 | 48.0% |
| delivery_pct | 0.0235 | -0.0104 | -0.22 | 48 | 94.3% |
| sector_mom | -0.017 | -0.1075 | -0.97 | 48 | 45.9% |
| bulk_deal | — | — | — | 0 | 0.7% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 94.3% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
