# Shadow Grade: untraded predictions vs reality

_Generated 2026-11-05 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **NO_EDGE**

Top-15 picks trailed NIFTY by -0.10% per 20-day hold after costs. The ranking is not adding value over simply holding the index.

- Signal days graded: 88 (2026-07-01 to 2026-11-04), prices through 2026-11-04, 236 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-15, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -5.15% | -10.45% |
| NIFTY, same timing | -0.79% | -9.95% |
| NIFTY buy and hold | -0.19% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-4.36%**

## Per-cohort (20-day holds)

- Matured cohorts: 66 (about 3 independent)
- Average excess vs NIFTY after costs: -0.1%
- Cohorts that beat NIFTY: 51.5%
- t-stat (overlap-adjusted): -0.06
- Top-15 minus bottom-15: 2.04% per hold, positive 72.7% of days (t=0.98)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| composite | 0.0439 | 0.0569 | 0.76 | 66 | — |
| momentum | 0.0274 | 0.0552 | 0.52 | 66 | 62.7% |
| mean_rev | 0.0323 | 0.034 | 0.39 | 66 | 32.9% |
| delivery_pct | 0.0348 | 0.0189 | 0.33 | 66 | 63.5% |
| hi52 | 0.0003 | -0.0097 | -0.12 | 66 | 48.4% |
| sector_mom | -0.0431 | -0.1101 | -1.24 | 66 | 29.7% |
| bulk_deal | — | — | — | 0 | 0.4% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 63.7% |
| fo_ban | — | — | — | 0 | 0.2% |
| mom_riskadj | — | — | — | 0 | 1.1% |
| pead | — | — | — | 0 | 0.3% |
| quality | — | — | — | 0 | 0.5% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
