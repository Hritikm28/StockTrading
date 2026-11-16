# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-16 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 0.87% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 95 (2026-07-01 to 2026-11-13), prices through 2026-11-16, 238 stocks priced
- Days the live trend gate allowed trading: 6

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -2.28% | -10.45% |
| NIFTY, same timing | 4.34% | -9.95% |
| NIFTY buy and hold | 5.35% | |
| What the gated live system did | -0.32% | |

Excess of picks over NIFTY (same timing): **-6.62%**

## Per-cohort (20-day holds)

- Matured cohorts: 72 (about 3 independent)
- Average excess vs NIFTY after costs: -0.87%
- Cohorts that beat NIFTY: 47.2%
- t-stat (overlap-adjusted): -0.43
- Top-15 minus bottom-15: 1.94% per hold, positive 70.8% of days (t=0.99)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0382 | 0.0491 | 0.7 | 72 | — |
| momentum | 0.0276 | 0.0476 | 0.48 | 72 | 62.0% |
| mean_rev | 0.0329 | 0.0377 | 0.45 | 72 | 33.4% |
| delivery_pct | 0.0338 | 0.0285 | 0.5 | 72 | 62.9% |
| hi52 | -0.0029 | -0.0162 | -0.21 | 72 | 47.9% |
| sector_mom | -0.0666 | -0.1182 | -1.4 | 72 | 29.3% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.1% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.046 | — | — | 0 | 5.4% |
| pead | 0.0661 | — | — | 0 | 3.0% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.1614 | — | — | 0 | 2.9% |
| resid_mom | — | — | — | 0 | 3.2% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
