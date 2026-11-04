# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-04 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by -0.07% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 88 (2026-07-01 to 2026-11-04), prices through 2026-11-04, 236 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.87% | -10.45% |
| NIFTY, same timing | -0.79% | -9.95% |
| NIFTY buy and hold | -0.19% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-3.08%**

## Per-cohort (20-day holds)

- Matured cohorts: 66 (about 3 independent)
- Average excess vs NIFTY after costs: -0.07%
- Cohorts that beat NIFTY: 51.5%
- t-stat (overlap-adjusted): -0.04
- Top-15 minus bottom-15: 2.05% per hold, positive 72.7% of days (t=0.99)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.044 | 0.0574 | 0.77 | 66 | — |
| momentum | 0.0286 | 0.0554 | 0.52 | 66 | 62.7% |
| mean_rev | 0.0336 | 0.0349 | 0.4 | 66 | 32.9% |
| delivery_pct | 0.0341 | 0.0188 | 0.32 | 66 | 63.5% |
| hi52 | -0.0008 | -0.0094 | -0.11 | 66 | 48.4% |
| sector_mom | -0.0467 | -0.11 | -1.24 | 66 | 29.7% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.7% |
| fo_ban | — | — | — | 0 | 0.2% |
| mom_riskadj | — | — | — | 0 | 1.1% |
| pead | — | — | — | 0 | 0.3% |
| quality | — | — | — | 0 | 0.5% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
