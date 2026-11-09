"""
What buying and selling shares really costs in India, and the tax on the
profit. Used by the paper portfolio (portfolio_book.py) and its reports.

Delivery trades (bought and held overnight or longer) through a discount
broker that charges no brokerage on delivery, at 2026 rates:

  - STT (securities transaction tax): 0.1% of the value, on buy and sell
  - stamp duty: 0.015% of the value, on buy only
  - NSE transaction charge: 0.00297%, plus 18% GST on it
  - SEBI fee: Rs 10 per crore (0.0001%), plus 18% GST
  - depository (DP) charge: about Rs 15.93 per stock on each day it is sold
  - slippage: paying about 0.1% worse than the quoted opening price

Roughly 0.22% to buy and 0.20% plus Rs 16 to sell.

Tax on profit from listed shares (since 23 July 2024):
  - held under 12 months: 20% (short-term capital gains)
  - held 12 months or more: 12.5% on gains above Rs 1.25 lakh a year
  - short-term losses can offset long-term gains, not the other way round
"""

STT_PCT = 0.10
STAMP_PCT = 0.015
EXCHANGE_PCT = 0.00297
SEBI_PCT = 0.0001
GST_RATE = 0.18
DP_CHARGE = 15.93
SLIPPAGE_PCT = 0.10

BUY_RATE = (STT_PCT + STAMP_PCT + (EXCHANGE_PCT + SEBI_PCT) * (1 + GST_RATE)
            + SLIPPAGE_PCT) / 100
SELL_RATE = (STT_PCT + (EXCHANGE_PCT + SEBI_PCT) * (1 + GST_RATE)
             + SLIPPAGE_PCT) / 100

STCG_RATE = 0.20
LTCG_RATE = 0.125
LTCG_EXEMPT = 125_000.0
LONG_TERM_DAYS = 365


def buy_cost(value: float) -> float:
    """Rupees paid on top of `value` to buy it."""
    return max(value, 0.0) * BUY_RATE


def sell_cost(value: float) -> float:
    """Rupees taken out of `value` when selling it."""
    return 0.0 if value <= 0 else value * SELL_RATE + DP_CHARGE


def capital_gains_tax(short_gain: float, long_gain: float) -> float:
    """Tax on one year's net gains. Short-term losses reduce long-term
    gains; long-term losses cannot reduce short-term gains."""
    st, lt = short_gain, long_gain
    if st < 0:
        lt += st
        st = 0.0
    return STCG_RATE * st + LTCG_RATE * max(lt - LTCG_EXEMPT, 0.0)
