# Factor Lab: ten-year test of the portfolio rules

_Generated 2026-11-09 by factor_lab.py. Period 2016-11-02 to 2026-11-09, 3269 stocks from NSE's daily bhav copies, including those that later collapsed or were delisted (no survivorship bias). Same rules as the live paper portfolio: top 15 (max 4 per industry), monthly replacement with a top-30 buffer, trend gate, 0.22% cost per side._

Price returns only, for the strategy and NIFTY alike. After-tax figures are estimates: strategy gains taxed at 20% each year, NIFTY held throughout and taxed 12.5% once at the end.

| Variant | CAGR | After tax | NIFTY | NIFTY after tax | Excess | Max fall | Sharpe | Turnover/yr | Months beating NIFTY | Chance it's skill |
|---|---|---|---|---|---|---|---|---|---|---|
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | 12.56% | 10.22% | 11.37% | 10.42% | **+1.19%** | -32.0% | 0.73 | 216% | 52.1% | 54.5% |
| Residual momentum, watch-list stocks allowed | 10.98% | 8.91% | 11.37% | 10.42% | **-0.39%** | -32.1% | 0.65 | 218% | 50.4% | 42.8% |
| Plain momentum 12-1, skipping watch-list stocks | 10.53% | 8.51% | 11.37% | 10.42% | **-0.84%** | -37.3% | 0.56 | 196% | 51.2% | 45.6% |
| Plain momentum 12-1 without the trend gate | 10.37% | 8.42% | 11.37% | 10.42% | **-1.0%** | -42.0% | 0.53 | 173% | 52.1% | 48.4% |
| Plain momentum 12-1 (live until 2026-11-09) | 9.88% | 8.01% | 11.37% | 10.42% | **-1.49%** | -36.2% | 0.54 | 194% | 49.6% | 41.2% |

**Survivorship bias, measured:** plain momentum tested the old way, on today's NIFTY 500 only (501 stocks), returned 20.73% a year (worst fall -38.7%), against 9.88% here. The old test overstated the yearly return by 10.8 points.

_"Chance it's skill" is the deflated Sharpe ratio of the monthly gap to NIFTY: the probability the edge is real after charging for every variant in this table. 95%+ is convincing; under 50% means luck explains it as well as skill. It does not charge for ideas tried before this lab existed, so read it as an upper bound too._

## Year by year (strategy % / NIFTY %)

| Variant | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Residual momentum, skipping watch-list stocks (live from 2026-11-10) | -15.3 / -3.9 | 33.2 / 28.6 | -7.3 / 3.2 | -5.7 / 12.0 | 18.0 / 14.9 | 47.4 / 24.1 | -4.1 / 4.3 | 37.2 / 20.0 | 38.2 / 8.8 | -1.2 / 10.5 | 6.2 / -4.2 |
| Residual momentum, watch-list stocks allowed | -15.3 / -3.9 | 33.2 / 28.6 | -5.2 / 3.2 | -5.8 / 12.0 | 16.8 / 14.9 | 41.6 / 24.1 | -9.7 / 4.3 | 35.7 / 20.0 | 40.5 / 8.8 | -4.6 / 10.5 | 3.9 / -4.2 |
| Plain momentum 12-1, skipping watch-list stocks | -17.3 / -3.9 | 46.8 / 28.6 | -19.9 / 3.2 | -1.5 / 12.0 | 13.1 / 14.9 | 39.3 / 24.1 | 8.0 / 4.3 | 19.5 / 20.0 | 31.0 / 8.8 | -8.4 / 10.5 | 16.6 / -4.2 |
| Plain momentum 12-1 (live until 2026-11-09) | -17.3 / -3.9 | 46.8 / 28.6 | -18.2 / 3.2 | -1.1 / 12.0 | 14.1 / 14.9 | 29.3 / 24.1 | 5.4 / 4.3 | 19.6 / 20.0 | 39.4 / 8.8 | -11.7 / 10.5 | 14.3 / -4.2 |
| Plain momentum 12-1 without the trend gate | -12.2 / -3.9 | 45.6 / 28.6 | -17.8 / 3.2 | 6.8 / 12.0 | 12.4 / 14.9 | 31.4 / 24.1 | -9.1 / 4.3 | 26.2 / 20.0 | 38.9 / 8.8 | -15.3 / 10.5 | 20.2 / -4.2 |

## The trend gate on NIFTY itself (no survivorship bias)

Holding NIFTY fully while it is above its 200-day average and cutting back when it falls 2% below (0.1% cost per switch), versus simply holding it. No stock list is involved, so this isolates what the gate adds or costs.

| | Yearly return | Worst fall |
|---|---|---|
| Gate: sell everything | 6.84% | -20.5% |
| Gate: sell half | 9.03% | -27.3% |
| NIFTY held throughout | 10.92% | -38.4% |

The gate switched 2.6 times a year.
