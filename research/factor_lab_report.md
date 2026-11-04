# Factor Lab: ten-year test of the portfolio rules

_Generated 2026-11-04 by factor_lab.py. Period 2016-10-28 to 2026-11-04, 501 candidate stocks (today's NIFTY 500). Same rules as the live paper portfolio: top 15 (max 4 per industry), monthly replacement with a top-30 buffer, trend gate, 0.4% round-trip costs._

**Read with care (survivorship bias):** the candidate list is today's NIFTY 500, so companies that collapsed and left the index are missing. That flatters every result below. Compare the variants with each other; treat the gap to NIFTY as an upper bound. Price returns only, for the strategy and NIFTY alike.

| Variant | CAGR | NIFTY | Excess | Max fall | Sharpe | Turnover/yr | Months beating NIFTY | Chance it's skill |
|---|---|---|---|---|---|---|---|---|
| Momentum 12-1 without the trend gate | 25.14% | 10.72% | **+14.42%** | -44.1% | 1.0 | 188% | 54.9% | 91.7% |
| Risk-adjusted momentum + 52-week high | 16.84% | 10.72% | **+6.12%** | -33.2% | 0.95 | 261% | 52.5% | 55.5% |
| Momentum 12-1 (live today) | 16.0% | 10.72% | **+5.28%** | -35.3% | 0.81 | 206% | 48.4% | 49.4% |
| Risk-adjusted momentum (NSE style) | 15.28% | 10.72% | **+4.56%** | -35.2% | 0.84 | 248% | 50.0% | 46.1% |
| Momentum, steady climbers preferred | 12.94% | 10.72% | **+2.22%** | -34.9% | 0.74 | 236% | 47.5% | 33.1% |
| Closeness to 52-week high | 12.57% | 10.72% | **+1.85%** | -31.9% | 0.83 | 537% | 44.3% | 28.2% |

_"Chance it's skill" is the deflated Sharpe ratio of the monthly gap to NIFTY: the probability the edge is real after charging for every variant in this table. 95%+ is convincing; under 50% means luck explains it as well as skill. It does not charge for ideas tried before this lab existed, so read it as an upper bound too._

## Year by year (strategy % / NIFTY %)

| Variant | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Momentum 12-1 (live today) | -23.2 / -5.2 | 54.2 / 28.6 | -6.2 / 3.2 | -5.3 / 12.0 | 29.7 / 14.9 | 75.7 / 24.1 | -9.6 / 4.3 | 48.1 / 20.0 | 50.3 / 8.8 | -3.7 / 10.5 | -4.8 / -8.3 |
| Risk-adjusted momentum (NSE style) | -22.9 / -5.2 | 57.4 / 28.6 | -9.4 / 3.2 | -3.8 / 12.0 | 32.9 / 14.9 | 74.6 / 24.1 | -2.5 / 4.3 | 59.7 / 20.0 | 32.0 / 8.8 | -12.0 / 10.5 | -6.6 / -8.3 |
| Momentum, steady climbers preferred | -23.1 / -5.2 | 50.2 / 28.6 | -9.3 / 3.2 | -5.7 / 12.0 | 37.8 / 14.9 | 73.1 / 24.1 | -7.6 / 4.3 | 31.6 / 20.0 | 32.3 / 8.8 | -6.8 / 10.5 | -4.3 / -8.3 |
| Closeness to 52-week high | -18.6 / -5.2 | 32.5 / 28.6 | -14.7 / 3.2 | -9.1 / 12.0 | 52.7 / 14.9 | 48.3 / 24.1 | -5.0 / 4.3 | 55.0 / 20.0 | 31.2 / 8.8 | -7.3 / 10.5 | -3.5 / -8.3 |
| Risk-adjusted momentum + 52-week high | -19.6 / -5.2 | 45.9 / 28.6 | -1.6 / 3.2 | -4.5 / 12.0 | 31.4 / 14.9 | 76.5 / 24.1 | 4.0 / 4.3 | 56.0 / 20.0 | 31.1 / 8.8 | -7.1 / 10.5 | -5.9 / -8.3 |
| Momentum 12-1 without the trend gate | -13.2 / -5.2 | 67.1 / 28.6 | 5.2 / 3.2 | 1.3 / 12.0 | 22.8 / 14.9 | 74.7 / 24.1 | 6.3 / 4.3 | 34.7 / 20.0 | 50.6 / 8.8 | -1.8 / 10.5 | 34.6 / -8.3 |

## The trend gate on NIFTY itself (no survivorship bias)

Holding NIFTY only while it is above its 200-day average (selling 2% below, 0.1% cost per switch), versus simply holding it. No stock list is involved, so this isolates what the gate adds or costs.

| | Yearly return | Worst fall |
|---|---|---|
| NIFTY with the gate | 6.61% | -20.5% |
| NIFTY held throughout | 10.41% | -38.4% |

The gate switched 2.5 times a year.
