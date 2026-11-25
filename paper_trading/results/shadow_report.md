# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-25 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.19% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 102 (2026-07-01 to 2026-11-24), prices through 2026-11-25, 240 stocks priced
- Days the live trend gate allowed trading: 13

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.0% | -10.26% |
| NIFTY, same timing | 6.48% | -9.95% |
| NIFTY buy and hold | 7.55% | |
| What the gated live system did | -1.42% | |

Excess of picks over NIFTY (same timing): **-9.48%**

## Per-cohort (20-day holds)

- Matured cohorts: 81 (about 4 independent)
- Average excess vs NIFTY after costs: -1.19%
- Cohorts that beat NIFTY: 44.4%
- t-stat (overlap-adjusted): -0.54
- Top-15 minus bottom-15: 1.67% per hold, positive 67.1% of days (t=0.88)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0186 | 0.0497 | 0.55 | 79 | 61.5% |
| composite | 0.0305 | 0.0437 | 0.67 | 79 | — |
| mean_rev | 0.0289 | 0.0406 | 0.53 | 79 | 33.1% |
| delivery_pct | 0.0442 | 0.0349 | 0.65 | 79 | 62.3% |
| hi52 | 0.0107 | -0.0024 | -0.03 | 79 | 47.5% |
| sector_mom | -0.0593 | -0.1272 | -1.57 | 79 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.6% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0049 | — | — | 0 | 9.2% |
| pead | 0.0196 | — | — | 0 | 6.5% |
| pledge | — | — | — | 0 | 0.0% |
| quality | -0.0022 | — | — | 0 | 6.9% |
| resid_mom | -0.0655 | — | — | 0 | 7.1% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
