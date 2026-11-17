# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-17 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 0.96% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 96 (2026-07-01 to 2026-11-16), prices through 2026-11-17, 238 stocks priced
- Days the live trend gate allowed trading: 7

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -2.4% | -10.45% |
| NIFTY, same timing | 4.74% | -9.95% |
| NIFTY buy and hold | 5.76% | |
| What the gated live system did | -0.41% | |

Excess of picks over NIFTY (same timing): **-7.14%**

## Per-cohort (20-day holds)

- Matured cohorts: 73 (about 3 independent)
- Average excess vs NIFTY after costs: -0.96%
- Cohorts that beat NIFTY: 46.6%
- t-stat (overlap-adjusted): -0.47
- Top-15 minus bottom-15: 1.92% per hold, positive 71.2% of days (t=0.99)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0382 | 0.0487 | 0.7 | 73 | — |
| momentum | 0.0281 | 0.0481 | 0.5 | 73 | 62.0% |
| mean_rev | 0.0312 | 0.0386 | 0.47 | 73 | 33.5% |
| delivery_pct | 0.0314 | 0.0297 | 0.53 | 73 | 62.9% |
| hi52 | -0.0011 | -0.0136 | -0.18 | 73 | 47.9% |
| sector_mom | -0.0712 | -0.1199 | -1.43 | 73 | 29.3% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.1% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0689 | — | — | 0 | 6.0% |
| pead | 0.0592 | — | — | 0 | 3.5% |
| pledge | — | — | — | 0 | 0.0% |
| quality | 0.0843 | — | — | 0 | 3.5% |
| resid_mom | 0.0682 | — | — | 0 | 3.8% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
