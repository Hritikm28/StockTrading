# Shadow Grade: untraded predictions vs reality

_Generated 2026-10-21 by shadow_grader.py. Grades every evening ranking since 2026-07-01, whether or not the trend gate let it trade. Entry at next-day open, 20-day holds, 0.4% round-trip costs._

## Verdict: **UNPROVEN**

Top-5 picks beat NIFTY by +0.87% per hold after costs, but t=0.42 is too weak to rule out luck.

- Signal days graded: 77 (2026-07-01 to 2026-10-20), prices through 2026-10-19, 158 stocks priced
- Days the live trend gate allowed trading: 0

## If we had traded every pick (top-5, ignoring the gate)

| | Return | Max drawdown |
|---|---|---|
| Shadow top picks (net) | -5.34% | -5.38% |
| NIFTY, same timing | -4.99% | -9.95% |
| NIFTY buy and hold | -4.65% | |
| What the gated live system did | 0.0% | |

Excess of picks over NIFTY (same timing): **-0.35%**

## Per-cohort (20-day holds)

- Matured cohorts: 54 (about 2 independent)
- Average excess vs NIFTY after costs: 0.87%
- Cohorts that beat NIFTY: 59.3%
- t-stat (overlap-adjusted): 0.42
- Top-5 minus bottom-5: 3.79% per hold, positive 74.1% of days (t=1.11)

## Which signals actually predict? (rank IC vs NIFTY-relative forward return)

| Signal | IC 5d | IC 20d | t 20d | days | coverage |
|---|---|---|---|---|---|
| momentum | 0.0352 | 0.0953 | 0.84 | 54 | 93.0% |
| composite | 0.0561 | 0.0795 | 0.94 | 54 | — |
| hi52 | -0.0152 | 0.0204 | 0.23 | 54 | 73.5% |
| mean_rev | 0.0506 | 0.0191 | 0.19 | 54 | 47.7% |
| delivery_pct | 0.0186 | -0.0128 | -0.27 | 54 | 94.2% |
| sector_mom | -0.027 | -0.1002 | -0.98 | 54 | 45.8% |
| bulk_deal | — | — | — | 0 | 0.7% |
| corp_event | — | — | — | 0 | 0.0% |
| fii_dii | — | — | — | 0 | 94.3% |
| fo_ban | — | — | — | 0 | 0.3% |

_IC of 0.02-0.05 is a tradeable edge, above 0.05 is strong, negative means the signal points the wrong way. |t| below 2 means the number could easily be luck._
