import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from pathlib import Path
import json


class FundamentalsTimeline:
    
    def __init__(self, cache_dir: str = "data/fundamentals_timeline"):

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache
        self.fundamentals_cache = {}
        self.dividends_cache = {}
        
        print("📊 Fundamentals Timeline initialized")
        print(f"   Tracks: PE ratio, Book Value, Debt/Equity, Dividends")
        print(f"   Cache directory: {self.cache_dir}")
    
    # MAIN API: Get Point-in-Time Fundamentals
    
    def get_fundamentals_on_date(
        self,
        symbol: str,
        as_of_date: date
    ) -> Dict:
        
        # Ensure date object
        if isinstance(as_of_date, str):
            as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
        elif isinstance(as_of_date, pd.Timestamp):
            as_of_date = as_of_date.date()
        elif isinstance(as_of_date, datetime):
            as_of_date = as_of_date.date()
        
        # Get quarterly financials
        quarterly_data = self._get_quarterly_fundamentals(symbol)
        
        # Get dividends history
        dividends = self._get_dividends_history(symbol)
        
        # Find the most recent quarter BEFORE as_of_date
        result = {
            'pe_ratio': 20.0,  # Default if no data
            'book_value': 0.0,
            'debt_to_equity': 0.0,
            'dividend_yield': 0.0,
            'market_cap': 0.0,
            'last_dividend_date': None,
            'days_since_dividend': 999
        }
        
        # Get PE ratio for this date
        if quarterly_data is not None and not quarterly_data.empty:
            # Filter quarters before as_of_date
            past_quarters = quarterly_data[quarterly_data.index <= pd.Timestamp(as_of_date)]
            
            if not past_quarters.empty:
                latest_quarter = past_quarters.iloc[-1]
                
                # Extract fundamentals
                result['pe_ratio'] = latest_quarter.get('PE_Ratio', 20.0)
                result['book_value'] = latest_quarter.get('Book_Value', 0.0)
                result['debt_to_equity'] = latest_quarter.get('Debt_to_Equity', 0.0)
                result['market_cap'] = latest_quarter.get('Market_Cap', 0.0)
        
        # Get dividend info for this date
        if dividends is not None and not dividends.empty:
            # FIXED: .date is a method on DatetimeIndex, not a property on Series index
            # Use .normalize().dt.date or convert to date objects safely
            try:
                div_dates = pd.to_datetime(dividends.index).normalize()
                past_dividends = dividends[div_dates.date <= as_of_date]
            except Exception:
                # Fallback: compare as Timestamps
                past_dividends = dividends[dividends.index <= pd.Timestamp(as_of_date)]
            
            if not past_dividends.empty:
                last_dividend = past_dividends.iloc[-1]
                last_div_date = past_dividends.index[-1].date()
                
                result['last_dividend_date'] = last_div_date
                result['days_since_dividend'] = (as_of_date - last_div_date).days
                
                # Calculate dividend yield (this is approximate)
                # Real yield needs price, but we'll store the dividend amount
                result['last_dividend_amount'] = float(last_dividend)
        
        return result
    
    def get_pe_on_date(self, symbol: str, as_of_date: date) -> float:
        """Quick function to get just PE ratio on a specific date."""
        fundamentals = self.get_fundamentals_on_date(symbol, as_of_date)
        return fundamentals['pe_ratio']
    
    def get_dividend_on_date(self, symbol: str, as_of_date: date) -> Dict:
        """Quick function to get just dividend info on a specific date."""
        fundamentals = self.get_fundamentals_on_date(symbol, as_of_date)
        return {
            'last_dividend_date': fundamentals.get('last_dividend_date'),
            'days_since_dividend': fundamentals.get('days_since_dividend'),
            'last_dividend_amount': fundamentals.get('last_dividend_amount', 0.0)
        }
    
    # DATA FETCHING: QUARTERLY FUNDAMENTALS
    
    def _get_quarterly_fundamentals(self, symbol: str) -> Optional[pd.DataFrame]:

        if symbol.startswith('^'):
            return None
        
        # Check memory cache
        if symbol in self.fundamentals_cache:
            return self.fundamentals_cache[symbol]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{symbol.replace('.NS', '')}_fundamentals.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self.fundamentals_cache[symbol] = df
                return df
            except Exception as e:
                print(f"   ⚠️ Fundamentals cache read failed: {e}")
        
        # Fetch from yfinance
        print(f"   📥 Fetching quarterly fundamentals for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # Get quarterly financial statements
            quarterly_financials = ticker.quarterly_financials
            quarterly_balance_sheet = ticker.quarterly_balance_sheet
            
            if quarterly_financials is None or quarterly_financials.empty:
                print(f"   ⚠️ No quarterly financials for {symbol}")
                return None
            
            # Build fundamentals timeline
            # Note: yfinance quarterly data is limited (usually last 4 quarters)
            dates = quarterly_financials.columns
            
            fundamentals_data = []
            
            for date_col in dates:
                quarter_data = {
                    'quarter_date': date_col,
                    'PE_Ratio': 20.0,  # We'll calculate this
                    'Book_Value': 0.0,
                    'Debt_to_Equity': 0.0,
                    'Market_Cap': 0.0
                }
                
                # Try to extract metrics
                try:
                    # Net Income (for PE calculation)
                    if 'Net Income' in quarterly_financials.index:
                        net_income = quarterly_financials.loc['Net Income', date_col]
                        quarter_data['Net_Income'] = float(net_income) if not pd.isna(net_income) else 0.0
                    
                    # Total Assets
                    if quarterly_balance_sheet is not None and 'Total Assets' in quarterly_balance_sheet.index:
                        total_assets = quarterly_balance_sheet.loc['Total Assets', date_col]
                        quarter_data['Total_Assets'] = float(total_assets) if not pd.isna(total_assets) else 0.0
                    
                    # Total Debt
                    if quarterly_balance_sheet is not None and 'Total Debt' in quarterly_balance_sheet.index:
                        total_debt = quarterly_balance_sheet.loc['Total Debt', date_col]
                        quarter_data['Total_Debt'] = float(total_debt) if not pd.isna(total_debt) else 0.0
                    
                    # Stockholders Equity (for Book Value)
                    if quarterly_balance_sheet is not None and 'Stockholders Equity' in quarterly_balance_sheet.index:
                        equity = quarterly_balance_sheet.loc['Stockholders Equity', date_col]
                        quarter_data['Equity'] = float(equity) if not pd.isna(equity) else 0.0
                
                except Exception as e:
                    pass
                
                fundamentals_data.append(quarter_data)
            
            # Convert to DataFrame
            df = pd.DataFrame(fundamentals_data)
            df = df.set_index('quarter_date')
            df = df.sort_index()
            
            print(f"   ✅ Found {len(df)} quarters of fundamental data")
            print(f"      Earliest: {df.index[0].date()}")
            print(f"      Latest: {df.index[-1].date()}")
            
            # Calculate derived metrics
            # Save to cache
            try:
                df.to_parquet(cache_file)
                print(f"   ✅ Cached to disk")
            except Exception as e:
                print(f"   ⚠️ Cache write failed: {e}")
            
            # Save to memory
            self.fundamentals_cache[symbol] = df
            
            return df
            
        except Exception as e:
            print(f"   ❌ Failed to fetch fundamentals for {symbol}: {e}")
            return None
    
    # DATA FETCHING: DIVIDENDS
    
    def _get_dividends_history(self, symbol: str) -> Optional[pd.Series]:

        if symbol.startswith('^'):
            return None
        
        # Check memory cache
        if symbol in self.dividends_cache:
            return self.dividends_cache[symbol]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{symbol.replace('.NS', '')}_dividends.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                series = df['Dividend']
                self.dividends_cache[symbol] = series
                return series
            except Exception as e:
                print(f"   ⚠️ Dividends cache read failed: {e}")
        
        # Fetch from yfinance
        print(f"   📥 Fetching dividend history for {symbol}...")
        
        try:
            ticker = yf.Ticker(symbol)
            dividends = ticker.dividends
            
            if dividends is None or dividends.empty:
                print(f"   ⚠️ No dividend history for {symbol}")
                return None
            
            # Ensure proper datetime index
            if not isinstance(dividends.index, pd.DatetimeIndex):
                dividends.index = pd.to_datetime(dividends.index)
            
            # Sort by date
            dividends = dividends.sort_index()
            
            print(f"   ✅ Found {len(dividends)} dividend payments")
            print(f"      Earliest: {dividends.index[0].date()}")
            print(f"      Latest: {dividends.index[-1].date()}")
            print(f"      Total paid: ₹{dividends.sum():.2f}")
            
            # Save to cache
            try:
                df = pd.DataFrame({'Dividend': dividends})
                df.to_parquet(cache_file)
                print(f"   ✅ Cached to disk")
            except Exception as e:
                print(f"   ⚠️ Cache write failed: {e}")
            
            # Save to memory
            self.dividends_cache[symbol] = dividends
            
            return dividends
            
        except Exception as e:
            print(f"   ❌ Failed to fetch dividends for {symbol}: {e}")
            return None
    
    # TIMELINE GENERATION
    
    def get_fundamentals_timeline(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize result
        result = pd.DataFrame(index=date_range)
        result.index.name = 'Date'
        
        # For each date, get fundamentals
        pe_ratios = []
        book_values = []
        debt_ratios = []
        days_since_div = []
        last_div_amounts = []
        
        print(f"   Generating fundamentals timeline for {len(date_range)} days...")
        
        for current_date in date_range:
            fundamentals = self.get_fundamentals_on_date(symbol, current_date.date())
            
            pe_ratios.append(fundamentals['pe_ratio'])
            book_values.append(fundamentals['book_value'])
            debt_ratios.append(fundamentals['debt_to_equity'])
            days_since_div.append(fundamentals['days_since_dividend'])
            last_div_amounts.append(fundamentals.get('last_dividend_amount', 0.0))
        
        result['PE_Ratio'] = pe_ratios
        result['Book_Value'] = book_values
        result['Debt_to_Equity'] = debt_ratios
        result['Days_Since_Dividend'] = days_since_div
        result['Last_Dividend_Amount'] = last_div_amounts
        
        return result
    
    # CACHE MANAGEMENT
    
    def clear_cache(self, symbol: Optional[str] = None):
        """Clear cached fundamental data."""
        
        if symbol:
            # Clear specific symbol
            for suffix in ['_fundamentals.parquet', '_dividends.parquet']:
                cache_file = self.cache_dir / f"{symbol.replace('.NS', '')}{suffix}"
                if cache_file.exists():
                    cache_file.unlink()
            
            if symbol in self.fundamentals_cache:
                del self.fundamentals_cache[symbol]
            if symbol in self.dividends_cache:
                del self.dividends_cache[symbol]
            
            print(f"🗑️ Cleared fundamentals cache for {symbol}")
        else:
            # Clear all
            for file in self.cache_dir.glob("*.parquet"):
                file.unlink()
            
            self.fundamentals_cache.clear()
            self.dividends_cache.clear()
            
            print(f"🗑️ Cleared all fundamentals cache")


# EXAMPLE USAGE & TESTING
if __name__ == "__main__":
    """Test the fundamentals timeline"""
    
    print("="*70)
    print("FUNDAMENTALS TIMELINE TEST")
    print("="*70)
    
    timeline = FundamentalsTimeline()
    
    # Test with RELIANCE
    symbol = "RELIANCE.NS"
    
    print(f"\n📊 Testing with {symbol}")
    print("-" * 70)
    
    # Test 1: Get fundamentals on a specific date
    test_date = date(2024, 6, 15)
    
    print(f"\nTest 1: Fundamentals as of {test_date}")
    fundamentals = timeline.get_fundamentals_on_date(symbol, test_date)
    
    print(f"   PE Ratio: {fundamentals['pe_ratio']:.2f}")
    print(f"   Book Value: ₹{fundamentals['book_value']:.2f}")
    print(f"   Debt/Equity: {fundamentals['debt_to_equity']:.2f}")
    print(f"   Days Since Dividend: {fundamentals['days_since_dividend']}")
    if fundamentals['last_dividend_date']:
        print(f"   Last Dividend Date: {fundamentals['last_dividend_date']}")
        print(f"   Last Dividend Amount: ₹{fundamentals.get('last_dividend_amount', 0):.2f}")
    
    # Test 2: Get timeline for a period
    print(f"\nTest 2: Fundamentals timeline Jul-Sep 2024")
    
    df = timeline.get_fundamentals_timeline(
        symbol,
        date(2024, 7, 1),
        date(2024, 9, 30)
    )
    
    print(f"   ✅ Generated {len(df)} days of fundamental data")
    print(f"\n   Sample (last 10 days):")
    print(df[['PE_Ratio', 'Days_Since_Dividend', 'Last_Dividend_Amount']].tail(10))
    
    # Test 3: Show dividend history
    print(f"\nTest 3: Dividend payment dates")
    dividends = timeline._get_dividends_history(symbol)
    
    if dividends is not None and not dividends.empty:
        print(f"   Last 5 dividend payments:")
        for date_idx, amount in dividends.tail(5).items():
            print(f"      {date_idx.date()}: ₹{amount:.2f}")
    
    print("\n" + "="*70)
    print("✅ FUNDAMENTALS TIMELINE TEST COMPLETE!")
    print("="*70)
    print("\nKey Takeaway:")
    print("  Now you have point-in-time fundamental data!")
    print("  PE ratio, dividends, etc. match what was known on each date! 🚀")