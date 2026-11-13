# Factor Lab: ten-year test of the portfolio rules

_Generated 2026-11-13 by factor_lab.py. Period 2016-11-08 to 2026-11-12, 3274 stocks from NSE's daily bhav copies, including those that later collapsed or were delisted (no survivorship bias). Same rules as the live paper portfolio: top 15 (max 4 per industry), monthly replacement with a top-30 buffer, trend gate, 0.22% cost per side._

Price returns only, for the strategy and NIFTY alike. After-tax figures are estimates: strategy gains taxed at 20% each year, NIFTY held throughout and taxed 12.5% once at the end.

| Variant | CAGR | After tax | NIFTY | NIFTY after tax | Excess | Max fall | Sharpe | Turnover/yr | Months beating NIFTY | Chance it's skill |
|---|---|---|---|---|---|---|---|---|---|---|
| Live rule with 20 stocks instead of 15 | 13.97% | 11.33% | 11.46% | 10.5% | **+2.51%** | -39.3% | 0.77 | 184% | 55.4% | 48.6% |
| Live rule, reviewed every 2 weeks instead of monthly | 13.54% | 11.03% | 11.46% | 10.5% | **+2.08%** | -35.3% | 0.76 | 235% | 50.4% | 44.4% |
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | 13.32% | 10.82% | 11.46% | 10.5% | **+1.87%** | -32.2% | 0.76 | 218% | 52.1% | 42.3% |
| Residual momentum, watch-list stocks allowed | 11.89% | 9.66% | 11.46% | 10.5% | **+0.44%** | -32.2% | 0.7 | 218% | 50.4% | 32.3% |
| Plain momentum 12-1 without the trend gate | 11.1% | 9.02% | 11.46% | 10.5% | **-0.35%** | -42.4% | 0.56 | 170% | 52.1% | 35.0% |
| Plain momentum 12-1, skipping watch-list stocks | 11.03% | 8.94% | 11.46% | 10.5% | **-0.43%** | -37.6% | 0.58 | 201% | 52.1% | 30.9% |
| Plain momentum 12-1 (live until 2026-11-09) | 10.08% | 8.19% | 11.46% | 10.5% | **-1.37%** | -36.6% | 0.55 | 200% | 51.2% | 25.0% |
| Live rule with 25 stocks instead of 15 | 9.32% | 7.45% | 11.46% | 10.5% | **-2.13%** | -35.6% | 0.57 | 172% | 44.6% | 18.1% |
| Live rule, blended with low volatility (steadier stocks) | 8.95% | 7.23% | 11.46% | 10.5% | **-2.51%** | -31.2% | 0.61 | 197% | 49.6% | 11.7% |

**Survivorship bias, measured:** plain momentum tested the old way, on today's NIFTY 500 only (501 stocks), returned 20.84% a year (worst fall -38.7%), against 10.08% here. The old test overstated the yearly return by 10.8 points.

## Risk detail

| Variant | Sortino | Return / worst fall | Worst 12 months | Longest time below a past peak | 3-year periods that beat NIFTY | Edge, first half | Edge, second half |
|---|---|---|---|---|---|---|---|
| Live rule with 20 stocks instead of 15 | 0.87 | 0.36 | -25.2% | 29 months | 64.7% | 0.14% | 4.76% |
| Live rule, reviewed every 2 weeks instead of monthly | 0.91 | 0.38 | -29.9% | 31 months | 68.2% | 0.46% | 3.61% |
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | 0.89 | 0.41 | -27.6% | 29 months | 63.5% | -2.8% | 6.37% |
| Residual momentum, watch-list stocks allowed | 0.82 | 0.37 | -27.6% | 31 months | 49.4% | -3.48% | 4.19% |
| Plain momentum 12-1 without the trend gate | 0.66 | 0.26 | -30.6% | 36 months | 35.3% | -2.95% | 2.1% |
| Plain momentum 12-1, skipping watch-list stocks | 0.67 | 0.29 | -28.1% | 31 months | 49.4% | -5.34% | 4.33% |
| Plain momentum 12-1 (live until 2026-11-09) | 0.64 | 0.28 | -28.0% | 32 months | 40.0% | -5.46% | 2.55% |
| Live rule with 25 stocks instead of 15 | 0.64 | 0.26 | -26.3% | 33 months | 54.1% | -4.0% | -0.38% |
| Live rule, blended with low volatility (steadier stocks) | 0.69 | 0.29 | -22.5% | 27 months | 21.2% | -2.25% | -2.74% |

_An edge that shows up in only one half of the period is more likely luck or a passing market phase. 'Steadier stocks' uses low price volatility as a stand-in for quality: profit and debt history isn't free for ten years, so real quality is graded live instead._

_"Chance it's skill" is the deflated Sharpe ratio of the monthly gap to NIFTY: the probability the edge is real after charging for every variant in this table. 95%+ is convincing; under 50% means luck explains it as well as skill. It does not charge for ideas tried before this lab existed, so read it as an upper bound too._

## Year by year (strategy % / NIFTY %)

| Variant | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | -7.6 / -4.2 | 30.7 / 28.6 | -6.6 / 3.2 | -5.8 / 12.0 | 18.0 / 14.9 | 47.4 / 24.1 | -4.1 / 4.3 | 35.9 / 20.0 | 36.6 / 8.8 | -1.2 / 10.5 | 7.5 / -3.2 |
| Live rule with 20 stocks instead of 15 | -8.8 / -4.2 | 37.4 / 28.6 | -8.9 / 3.2 | -3.5 / 12.0 | 24.3 / 14.9 | 51.3 / 24.1 | 11.3 / 4.3 | 9.4 / 20.0 | 39.8 / 8.8 | -3.7 / 10.5 | 9.0 / -3.2 |
| Live rule with 25 stocks instead of 15 | -8.6 / -4.2 | 31.1 / 28.6 | -18.0 / 3.2 | -1.1 / 12.0 | 22.7 / 14.9 | 42.0 / 24.1 | 9.0 / 4.3 | 9.8 / 20.0 | 27.6 / 8.8 | -8.4 / 10.5 | 3.0 / -3.2 |
| Live rule, reviewed every 2 weeks instead of monthly | -7.3 / -4.2 | 39.8 / 28.6 | -15.1 / 3.2 | -4.5 / 12.0 | 18.8 / 14.9 | 62.3 / 24.1 | -13.8 / 4.3 | 38.6 / 20.0 | 38.8 / 8.8 | -1.9 / 10.5 | 8.2 / -3.2 |
| Live rule, blended with low volatility (steadier stocks) | -6.6 / -4.2 | 28.7 / 28.6 | -4.4 / 3.2 | 5.0 / 12.0 | 22.6 / 14.9 | 25.9 / 24.1 | -18.4 / 4.3 | 21.9 / 20.0 | 29.1 / 8.8 | -5.4 / 10.5 | 4.4 / -3.2 |
| Residual momentum, watch-list stocks allowed | -7.6 / -4.2 | 30.7 / 28.6 | -4.5 / 3.2 | -5.9 / 12.0 | 16.8 / 14.9 | 41.2 / 24.1 | -9.8 / 4.3 | 35.5 / 20.0 | 40.5 / 8.8 | -4.6 / 10.5 | 5.0 / -3.2 |
| Plain momentum 12-1, skipping watch-list stocks | -8.9 / -4.2 | 45.6 / 28.6 | -19.2 / 3.2 | -4.2 / 12.0 | 14.8 / 14.9 | 29.6 / 24.1 | 1.6 / 4.3 | 24.4 / 20.0 | 38.7 / 8.8 | -9.7 / 10.5 | 18.0 / -3.2 |
| Plain momentum 12-1 (live until 2026-11-09) | -8.9 / -4.2 | 45.6 / 28.6 | -17.6 / 3.2 | -4.1 / 12.0 | 15.4 / 14.9 | 26.1 / 24.1 | -3.2 / 4.3 | 27.1 / 20.0 | 37.8 / 8.8 | -12.2 / 10.5 | 15.1 / -3.2 |
| Plain momentum 12-1 without the trend gate | -2.7 / -4.2 | 46.6 / 28.6 | -17.7 / 3.2 | 6.1 / 12.0 | 10.4 / 14.9 | 30.1 / 24.1 | -11.8 / 4.3 | 26.4 / 20.0 | 35.4 / 8.8 | -7.7 / 10.5 | 15.2 / -3.2 |

## The trend gate on NIFTY itself (no survivorship bias)

Holding NIFTY fully while it is above its 200-day average and cutting back when it falls 2% below (0.1% cost per switch), versus simply holding it. No stock list is involved, so this isolates what the gate adds or costs.

| | Yearly return | Worst fall |
|---|---|---|
| Gate: sell everything | 6.89% | -20.5% |
| Gate: sell half | 9.09% | -27.3% |
| NIFTY held throughout | 10.97% | -38.4% |

The gate switched 2.6 times a year.
