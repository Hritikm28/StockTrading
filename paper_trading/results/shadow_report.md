# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-16 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.94% per hold after costs, but t=0.44 is too weak to rule out luck.

- Signal days graded: 74 (2026-07-01 to 2026-10-15), prices through 2026-10-15, 158 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.85% | -5.04% |
| NIFTY, same timing | -4.55% | -9.95% |
| NIFTY buy and hold | -4.2% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **0.7%**

## Per-cohort (20-day holds)

- Matured cohorts: 52 (about 2 independent)
- Average excess vs NIFTY after costs: 0.94%
- Cohorts that beat NIFTY: 61.5%
- t-stat (overlap-adjusted): 0.44
- Top-5 minus bottom-5: 3.81% per hold, positive 73.1% of days (t=1.08)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0398 | 0.1055 | 0.94 | 52 | 93.0% |
| composite | 0.0609 | 0.0857 | 1.01 | 52 | — |
| hi52 | -0.01 | 0.0299 | 0.35 | 52 | 74.0% |
| mean_rev | 0.0467 | 0.0168 | 0.17 | 52 | 47.7% |
| delivery_pct | 0.0196 | -0.0147 | -0.32 | 52 | 94.2% |
| sector_mom | -0.0249 | -0.0988 | -0.93 | 52 | 45.8% |
| bulk_deal | — | — | — | 0 | 0.7% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 94.3% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
