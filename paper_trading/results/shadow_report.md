# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-24 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by 1.39% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 101 (2026-07-01 to 2026-11-23), prices through 2026-11-23, 240 stocks priced
- Days the live trend gate allowed trading: 12

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -3.62% | -10.45% |
| NIFTY, same timing | 4.61% | -9.95% |
| NIFTY buy and hold | 5.6% | |
| What the gated live system did | -1.13% | |

Excess of picks over NIFTY (same timing): **-8.23%**

## Per-cohort (20-day holds)

- Matured cohorts: 78 (about 3 independent)
- Average excess vs NIFTY after costs: -1.39%
- Cohorts that beat NIFTY: 43.6%
- t-stat (overlap-adjusted): -0.65
- Top-15 minus bottom-15: 1.73% per hold, positive 67.9% of days (t=0.9)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0222 | 0.0501 | 0.55 | 78 | 61.5% |
| composite | 0.0333 | 0.0451 | 0.68 | 78 | — |
| mean_rev | 0.0291 | 0.0382 | 0.49 | 78 | 33.1% |
| delivery_pct | 0.0412 | 0.0353 | 0.65 | 78 | 62.3% |
| hi52 | 0.0095 | -0.0027 | -0.04 | 78 | 47.5% |
| sector_mom | -0.0588 | -0.1276 | -1.56 | 78 | 29.2% |
| bulk_deal | — | — | — | 0 | 0.5% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 62.6% |
| fo_ban | — | — | — | 0 | 0.2% |
| insider | — | — | — | 0 | 0.0% |
| mom_riskadj | 0.0269 | — | — | 0 | 8.7% |
| pead | 0.0367 | — | — | 0 | 6.0% |
| pledge | — | — | — | 0 | 0.0% |
| quality | -0.0046 | — | — | 0 | 6.4% |
| resid_mom | -0.0334 | — | — | 0 | 6.5% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
