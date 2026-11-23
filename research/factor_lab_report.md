# Factor Lab: ten-year test of the portfolio rules

_Generated 2026-11-23 by factor_lab.py. Period 2016-11-17 to 2026-11-23, 3279 stocks from NSE's daily bhav copies, including those that later collapsed or were delisted (no survivorship bias). Same rules as the live paper portfolio: top 15 (max 4 per industry), monthly replacement with a top-30 buffer, trend gate, 0.22% cost per side._

Price returns only, for the strategy and NIFTY alike. After-tax figures are estimates: strategy gains taxed at 20% each year, NIFTY held throughout and taxed 12.5% once at the end.

| Variant | CAGR | After tax | NIFTY | NIFTY after tax | Excess | Max fall | Sharpe | Turnover/yr | Months beating NIFTY | Chance it's skill |
|---|---|---|---|---|---|---|---|---|---|---|
| Live rule with 20 stocks instead of 15 | 14.36% | 11.65% | 12.09% | 11.1% | **+2.27%** | -39.7% | 0.8 | 182% | 56.2% | 48.3% |
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | 13.7% | 11.13% | 12.09% | 11.1% | **+1.6%** | -32.2% | 0.79 | 214% | 52.1% | 41.8% |
| Live rule, reviewed every 2 weeks instead of monthly | 12.42% | 10.09% | 12.09% | 11.1% | **+0.32%** | -29.7% | 0.74 | 233% | 54.5% | 32.0% |
| Plain momentum 12-1, skipping watch-list stocks | 12.16% | 9.88% | 12.09% | 11.1% | **+0.07%** | -37.9% | 0.63 | 196% | 53.7% | 35.5% |
| Residual momentum, watch-list stocks allowed | 12.12% | 9.82% | 12.09% | 11.1% | **+0.03%** | -32.3% | 0.72 | 215% | 50.4% | 30.9% |
| Plain momentum 12-1 without the trend gate | 12.07% | 9.83% | 12.09% | 11.1% | **-0.02%** | -43.0% | 0.59 | 171% | 52.9% | 38.6% |
| Plain momentum 12-1 (live until 2026-11-09) | 10.97% | 8.89% | 12.09% | 11.1% | **-1.12%** | -36.6% | 0.59 | 195% | 51.2% | 27.8% |
| Live rule, blended with low volatility (steadier stocks) | 9.59% | 7.73% | 12.09% | 11.1% | **-2.5%** | -31.5% | 0.65 | 194% | 50.4% | 12.6% |
| Live rule with 25 stocks instead of 15 | 9.57% | 7.64% | 12.09% | 11.1% | **-2.52%** | -35.1% | 0.59 | 169% | 43.8% | 17.0% |

**Survivorship bias, measured:** plain momentum tested the old way, on today's NIFTY 500 only (501 stocks), returned 22.19% a year (worst fall -39.2%), against 10.97% here. The old test overstated the yearly return by 11.2 points.

## Risk detail

| Variant | Sortino | Return / worst fall | Worst 12 months | Longest time below a past peak | 3-year periods that beat NIFTY | Edge, first half | Edge, second half |
|---|---|---|---|---|---|---|---|
| Live rule with 20 stocks instead of 15 | 0.9 | 0.36 | -25.5% | 29 months | 64.7% | 0.19% | 4.23% |
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | 0.93 | 0.43 | -27.6% | 29 months | 63.5% | -2.98% | 6.02% |
| Live rule, reviewed every 2 weeks instead of monthly | 0.88 | 0.42 | -26.0% | 32 months | 50.6% | -2.74% | 3.24% |
| Plain momentum 12-1, skipping watch-list stocks | 0.73 | 0.32 | -28.0% | 39 months | 50.6% | -4.71% | 4.69% |
| Residual momentum, watch-list stocks allowed | 0.85 | 0.38 | -27.7% | 31 months | 49.4% | -3.71% | 3.6% |
| Plain momentum 12-1 without the trend gate | 0.71 | 0.28 | -29.9% | 37 months | 37.6% | -2.13% | 1.97% |
| Plain momentum 12-1 (live until 2026-11-09) | 0.69 | 0.3 | -27.9% | 31 months | 44.7% | -4.86% | 2.45% |
| Live rule, blended with low volatility (steadier stocks) | 0.75 | 0.3 | -22.4% | 27 months | 22.4% | -1.67% | -3.25% |
| Live rule with 25 stocks instead of 15 | 0.66 | 0.27 | -25.9% | 33 months | 54.1% | -4.18% | -0.97% |

_An edge that shows up in only one half of the period is more likely luck or a passing market phase. 'Steadier stocks' uses low price volatility as a stand-in for quality: profit and debt history isn't free for ten years, so real quality is graded live instead._

_"Chance it's skill" is the deflated Sharpe ratio of the monthly gap to NIFTY: the probability the edge is real after charging for every variant in this table. 95%+ is convincing; under 50% means luck explains it as well as skill. It does not charge for ideas tried before this lab existed, so read it as an upper bound too._

## Year by year (strategy % / NIFTY %)

| Variant | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | -3.7 / 1.3 | 31.2 / 28.6 | -6.9 / 3.2 | -5.8 / 12.0 | 18.0 / 14.9 | 47.4 / 24.1 | -4.1 / 4.3 | 36.0 / 20.0 | 36.6 / 8.8 | -1.2 / 10.5 | 6.5 / -3.0 |
| Live rule with 20 stocks instead of 15 | -4.3 / 1.3 | 36.9 / 28.6 | -8.4 / 3.2 | -3.7 / 12.0 | 25.7 / 14.9 | 50.2 / 24.1 | 11.3 / 4.3 | 9.2 / 20.0 | 39.8 / 8.8 | -3.7 / 10.5 | 7.4 / -3.0 |
| Live rule with 25 stocks instead of 15 | -4.0 / 1.3 | 30.0 / 28.6 | -17.9 / 3.2 | -0.9 / 12.0 | 21.8 / 14.9 | 40.9 / 24.1 | 10.2 / 4.3 | 8.9 / 20.0 | 27.8 / 8.8 | -8.4 / 10.5 | 2.1 / -3.0 |
| Live rule, reviewed every 2 weeks instead of monthly | -3.6 / 1.3 | 50.7 / 28.6 | -16.4 / 3.2 | -0.1 / 12.0 | 13.3 / 14.9 | 35.9 / 24.1 | -5.9 / 4.3 | 25.5 / 20.0 | 34.7 / 8.8 | 0.1 / 10.5 | 8.6 / -3.0 |
| Live rule, blended with low volatility (steadier stocks) | -3.5 / 1.3 | 32.0 / 28.6 | -1.9 / 3.2 | 4.7 / 12.0 | 22.0 / 14.9 | 25.3 / 24.1 | -18.3 / 4.3 | 21.8 / 20.0 | 28.7 / 8.8 | -5.3 / 10.5 | 3.3 / -3.0 |
| Residual momentum, watch-list stocks allowed | -3.7 / 1.3 | 31.2 / 28.6 | -4.8 / 3.2 | -5.9 / 12.0 | 16.7 / 14.9 | 41.2 / 24.1 | -9.8 / 4.3 | 35.5 / 20.0 | 40.5 / 8.8 | -4.6 / 10.5 | 2.8 / -3.0 |
| Plain momentum 12-1, skipping watch-list stocks | -4.4 / 1.3 | 51.7 / 28.6 | -19.7 / 3.2 | -3.9 / 12.0 | 14.1 / 14.9 | 31.0 / 24.1 | 3.4 / 4.3 | 23.0 / 20.0 | 38.6 / 8.8 | -9.2 / 10.5 | 17.9 / -3.0 |
| Plain momentum 12-1 (live until 2026-11-09) | -4.4 / 1.3 | 51.7 / 28.6 | -18.0 / 3.2 | -3.8 / 12.0 | 14.8 / 14.9 | 27.6 / 24.1 | -0.6 / 4.3 | 25.7 / 20.0 | 38.5 / 8.8 | -12.3 / 10.5 | 11.6 / -3.0 |
| Plain momentum 12-1 without the trend gate | 0.2 / 1.3 | 52.7 / 28.6 | -18.3 / 3.2 | 5.8 / 12.0 | 12.8 / 14.9 | 31.0 / 24.1 | -14.1 / 4.3 | 30.0 / 20.0 | 36.7 / 8.8 | -7.4 / 10.5 | 13.2 / -3.0 |

## The trend gate on NIFTY itself (no survivorship bias)

Holding NIFTY fully while it is above its 200-day average and cutting back when it falls 2% below (0.1% cost per switch), versus simply holding it. No stock list is involved, so this isolates what the gate adds or costs.

| | Yearly return | Worst fall |
|---|---|---|
| Gate: sell everything | 6.99% | -20.5% |
| Gate: sell half | 9.18% | -27.3% |
| NIFTY held throughout | 11.07% | -38.4% |

The gate switched 2.6 times a year.
