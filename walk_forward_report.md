# Walk-Forward Validation Report

_Generated 2026-06-11. Yearly retrain on strictly prior data; top-10 by probability (gate 0.55), 5-day holds, 0.4% round-trip costs. Event alphas (FII/DII, pledge, SAST...) are NOT in this backtest — no free point-in-time history exists; they are validated by the live track record instead._

## Overall (2020-2026)
- Strategy total: **-3.6%**  |  NIFTY same periods: **+52.7%**
- Avg excess per 5d period: -0.305%  |  period win rate: 54%  |  Sharpe (net): 0.03  |  max drawdown: -32.3%

| Year | Val AUC | Periods | Invested | Strategy | NIFTY | Excess |
|------|---------|---------|----------|----------|-------|--------|
| 2020 | 0.540 | 51 | 39% | +0.0% | +16.6% | **-16.6%** |
| 2021 | 0.523 | 50 | 32% | +15.8% | +27.9% | **-12.1%** |
| 2022 | 0.566 | 50 | 54% | -16.8% | +2.4% | **-19.2%** |

## Price-alpha rank ICs (mean per-date Spearman vs 5d fwd return)

| Factor | IC | Read |
|--------|----|------|
| momentum_12_1 | +0.0262 | real edge |
| mean_rev (RSI5 inv) | +0.0162 | weak |
| rel_strength_63d | -0.0111 | NEGATIVE — candidate to kill |
| low_vol (vol21 inv) | +0.0095 | weak |

_Rule of thumb: |IC| of 0.02-0.05 is a tradeable edge; 0.05+ is excellent; negative means the factor as signed loses money at this horizon._