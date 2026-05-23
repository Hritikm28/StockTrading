from enum import Enum
from typing import Dict


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class TradeType(Enum):
    """Trade classification"""
    INTRADAY = "intraday"
    DELIVERY = "delivery"


class IndianTransactionCosts:
    
    # Constants (all in percentage unless specified)
    BROKERAGE_FLAT = 20  # Rs per order for discount brokers
    BROKERAGE_PCT_INTRADAY = 0.03 / 100  # 0.03%
    BROKERAGE_PCT_DELIVERY = 0.03 / 100  # 0.03% (some brokers offer 0%)
    
    STT_INTRADAY_SELL = 0.025 / 100  # 0.025% only on sell side
    STT_DELIVERY_BUY_SELL = 0.1 / 100  # 0.1% on both sides
    
    TRANSACTION_CHARGES = 0.00325 / 100  # 0.00325% (NSE charges)
    SEBI_CHARGES = 0.0001 / 100  # 0.0001% (₹10 per crore)
    
    STAMP_DUTY_INTRADAY = 0.003 / 100  # 0.003% on buy side
    STAMP_DUTY_DELIVERY = 0.015 / 100  # 0.015% on buy side
    
    GST_RATE = 0.18  # 18% on (brokerage + transaction charges)
    
    @staticmethod
    def calculate_costs(
        trade_value: float,
        side: OrderSide,
        trade_type: TradeType,
        use_flat_brokerage: bool = True
    ) -> Dict[str, float]:
        
        costs = {}
        
        # 1. Brokerage
        if use_flat_brokerage:
            # FIXED: Use max() not min() — flat ₹20 is the FLOOR, not the cap
            # Discount brokers charge HIGHER of ₹20 flat OR percentage
            brokerage = max(IndianTransactionCosts.BROKERAGE_FLAT,
                           trade_value * IndianTransactionCosts.BROKERAGE_PCT_DELIVERY)
        else:
            if trade_type == TradeType.INTRADAY:
                brokerage = trade_value * IndianTransactionCosts.BROKERAGE_PCT_INTRADAY
            else:
                brokerage = trade_value * IndianTransactionCosts.BROKERAGE_PCT_DELIVERY
        
        costs['brokerage'] = brokerage
        
        # 2. STT (Securities Transaction Tax)
        if trade_type == TradeType.INTRADAY:
            if side == OrderSide.SELL:
                stt = trade_value * IndianTransactionCosts.STT_INTRADAY_SELL
            else:
                stt = 0
        else:  # DELIVERY
            stt = trade_value * IndianTransactionCosts.STT_DELIVERY_BUY_SELL
        
        costs['stt'] = stt
        
        # 3. Transaction charges (NSE)
        transaction_charges = trade_value * IndianTransactionCosts.TRANSACTION_CHARGES
        costs['transaction_charges'] = transaction_charges
        
        # 4. GST (on brokerage + transaction charges)
        gst_base = brokerage + transaction_charges
        gst = gst_base * IndianTransactionCosts.GST_RATE
        costs['gst'] = gst
        
        # 5. SEBI charges
        sebi = trade_value * IndianTransactionCosts.SEBI_CHARGES
        costs['sebi'] = sebi
        
        # 6. Stamp duty (only on buy side)
        if side == OrderSide.BUY:
            if trade_type == TradeType.INTRADAY:
                stamp = trade_value * IndianTransactionCosts.STAMP_DUTY_INTRADAY
            else:
                stamp = trade_value * IndianTransactionCosts.STAMP_DUTY_DELIVERY
        else:
            stamp = 0
        
        costs['stamp_duty'] = stamp
        
        # 7. Total
        total = sum(costs.values())
        costs['total'] = total
        costs['total_pct'] = (total / trade_value * 100) if trade_value > 0 else 0
        
        return costs
    
    @staticmethod
    def calculate_round_trip_cost(
        trade_value: float,
        trade_type: TradeType,
        use_flat_brokerage: bool = True
    ) -> float:

        buy_costs = IndianTransactionCosts.calculate_costs(
            trade_value, OrderSide.BUY, trade_type, use_flat_brokerage
        )
        sell_costs = IndianTransactionCosts.calculate_costs(
            trade_value, OrderSide.SELL, trade_type, use_flat_brokerage
        )
        
        return buy_costs['total'] + sell_costs['total']
    
    @staticmethod
    def print_cost_breakdown(trade_value: float, trade_type: TradeType):

        print(f"\n{'='*60}")
        print(f"TRANSACTION COST BREAKDOWN ({trade_type.value.upper()})")
        print(f"{'='*60}")
        print(f"Trade Value: ₹{trade_value:,.2f}")
        print(f"\nBUY SIDE:")
        print(f"{'-'*60}")
        
        buy_costs = IndianTransactionCosts.calculate_costs(
            trade_value, OrderSide.BUY, trade_type, True
        )
        
        for key, value in buy_costs.items():
            if key != 'total_pct':
                print(f"  {key.replace('_', ' ').title():.<40} ₹{value:>10.2f}")
        
        print(f"\nSELL SIDE:")
        print(f"{'-'*60}")
        
        sell_costs = IndianTransactionCosts.calculate_costs(
            trade_value, OrderSide.SELL, trade_type, True
        )
        
        for key, value in sell_costs.items():
            if key != 'total_pct':
                print(f"  {key.replace('_', ' ').title():.<40} ₹{value:>10.2f}")
        
        round_trip = buy_costs['total'] + sell_costs['total']
        round_trip_pct = (round_trip / trade_value) * 100
        
        print(f"\n{'='*60}")
        print(f"ROUND TRIP TOTAL: ₹{round_trip:,.2f} ({round_trip_pct:.3f}%)")
        print(f"{'='*60}\n")


# Example usage
if __name__ == "__main__":
    # Example: ₹1 lakh trade
    IndianTransactionCosts.print_cost_breakdown(100000, TradeType.DELIVERY)
    IndianTransactionCosts.print_cost_breakdown(100000, TradeType.INTRADAY)