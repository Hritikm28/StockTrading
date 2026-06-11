# Walk-Forward Validation Report

_Generated 2026-06-11. Yearly retrain on strictly prior data; top-10 by predicted cross-sectional rank, 5-day holds, 0.4% round-trip costs. Event alphas (FII/DII, pledge, SAST...) are NOT in this backtest — no free point-in-time history exists; they are validated by the live track record instead._

## Overall (2020-2026)
- Strategy total: **-20.3%**  |  NIFTY same periods: **+38.0%**
- Avg excess per 5d period: -0.235%  |  period win rate: 49%  |  Sharpe (net): -0.18  |  max drawdown: -32.0%

| Year | Val IC | Periods | Invested | Strategy | NIFTY | Excess |
|------|--------|---------|----------|----------|-------|--------|
| 2022 | +0.028 | 50 | 100% | -18.5% | +2.4% | **-20.8%** |
| 2023 | +0.018 | 49 | 100% | +4.4% | +19.5% | **-15.1%** |
| 2024 | +0.027 | 50 | 100% | +12.2% | +9.0% | **+3.2%** |
| 2025 | +0.034 | 50 | 100% | -8.0% | +10.9% | **-18.9%** |
| 2026 | +0.040 | 18 | 100% | -9.3% | -6.7% | **-2.6%** |

## Price-alpha rank ICs (mean per-date Spearman vs 5d fwd return)

| Factor | IC | Read |
|--------|----|------|
| momentum_12_1 | +0.0228 | real edge |
| mean_rev (RSI5 inv) | +0.0207 | real edge |
| rel_strength_63d | -0.0016 | NEGATIVE — candidate to kill |
| low_vol (vol21 inv) | +0.0063 | weak |

_Rule of thumb: |IC| of 0.02-0.05 is a tradeable edge; 0.05+ is excellent; negative means the factor as signed loses money at this horizon._