# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-12 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 0.67% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 93 (2026-07-01 to 2026-11-11), prices through 2026-11-12, 238 stocks priced
- Days the live trend gate allowed trading: 4

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -1.69% | -10.45% |
| NIFTY, same timing | 4.37% | -9.95% |
| NIFTY buy and hold | 5.39% | |
| What the gated live system did | 0.11% | |

Excess of picks over NIFTY (same timing): **-6.06%**

## Per-cohort (20-day holds)

- Matured cohorts: 70 (about 3 independent)
- Average excess vs NIFTY after costs: -0.67%
- Cohorts that beat NIFTY: 48.6%
- t-stat (overlap-adjusted): -0.34
- Top-15 minus bottom-15: 1.93% per hold, positive 71.4% of days (t=0.97)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0374 | 0.051 | 0.71 | 70 | — |
| momentum | 0.0264 | 0.0481 | 0.48 | 70 | 62.1% |
| mean_rev | 0.0316 | 0.0367 | 0.43 | 70 | 33.4% |
| delivery_pct | 0.0369 | 0.0264 | 0.46 | 70 | 62.9% |
| hi52 | -0.0043 | -0.0173 | -0.22 | 70 | 47.9% |
| sector_mom | -0.0567 | -0.1151 | -1.34 | 70 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.1% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | -0.0087 | — | — | 0 | 4.2% |
| pead | 0.0084 | — | — | 0 | 2.1% |
| pledge | — | — | — | 0 | 0.0% |
| quality | -0.0641 | — | — | 0 | 1.9% |
| resid_mom | — | — | — | 0 | 1.9% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
