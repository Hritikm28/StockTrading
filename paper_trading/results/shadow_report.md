# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-19 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.29% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 98 (2026-07-01 to 2026-11-18), prices through 2026-11-19, 239 stocks priced
- Days the live trend gate allowed trading: 9

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -1.92% | -10.45% |
| NIFTY, same timing | 5.29% | -9.95% |
| NIFTY buy and hold | 6.28% | |
| What the gated live system did | -0.08% | |

Excess of picks over NIFTY (same timing): **-7.21%**

## Per-cohort (20-day holds)

- Matured cohorts: 76 (about 3 independent)
- Average excess vs NIFTY after costs: -1.29%
- Cohorts that beat NIFTY: 44.7%
- t-stat (overlap-adjusted): -0.6
- Top-15 minus bottom-15: 1.72% per hold, positive 68.4% of days (t=0.89)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0263 | 0.0485 | 0.52 | 76 | 61.8% |
| composite | 0.0367 | 0.0451 | 0.67 | 76 | — |
| mean_rev | 0.0299 | 0.0364 | 0.46 | 76 | 33.3% |
| delivery_pct | 0.0355 | 0.0324 | 0.6 | 76 | 62.6% |
| hi52 | 0.0052 | -0.0062 | -0.08 | 76 | 47.7% |
| sector_mom | -0.0661 | -0.1241 | -1.51 | 76 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.8% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0485 | — | — | 0 | 7.1% |
| pead | 0.0501 | — | — | 0 | 4.5% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.0281 | — | — | 0 | 4.7% |
| resid_mom | 0.0172 | — | — | 0 | 4.9% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
