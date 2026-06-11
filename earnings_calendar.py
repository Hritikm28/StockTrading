import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from pathlib import Path
import json


class EarningsCalendar:
    
    def __init__(self, cache_dir: str = "data/earnings_calendar"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory cache for speed
        self.earnings_cache = {}
        
        print("📅 Earnings Calendar initialized (FIXED VERSION)")
        print(f"   Using earnings_dates (actual announcement dates)")
        print(f"   Cache directory: {self.cache_dir}")
    
    def get_last_announcement_before(
        self, 
        symbol: str, 
        as_of_date: date
    ) -> Optional[Dict]:
        
        # Ensure date object
        if isinstance(as_of_date, str):
            as_of_date = datetime.strptime(as_of_date, '%Y-%m-%d').date()
        elif isinstance(as_of_date, pd.Timestamp):
            as_of_date = as_of_date.date()
        elif isinstance(as_of_date, datetime):
            as_of_date = as_of_date.date()
        
        # Get earnings history for this symbol
        earnings_history = self._get_earnings_history(symbol)
        
        if earnings_history is None or earnings_history.empty:
            return None
        
        # ✅ FIX: Convert index to date for comparison
        earnings_dates = [idx.date() if hasattr(idx, 'date') else idx for idx in earnings_history.index]
        
        # Filter: only announcements BEFORE as_of_date
        mask = [d <= as_of_date for d in earnings_dates]
        past_earnings = earnings_history[mask]
        
        if past_earnings.empty:
            return None
        
        # Get the MOST RECENT announcement before as_of_date
        last_earnings = past_earnings.iloc[-1]
        last_date = past_earnings.index[-1]
        
        # Extract EPS data
        eps_actual = last_earnings.get('Reported EPS', 0)
        eps_estimate = last_earnings.get('EPS Estimate', 0)
        
        # Handle NaN values
        if pd.isna(eps_actual):
            eps_actual = 0
        if pd.isna(eps_estimate):
            eps_estimate = 0
        
        # Calculate surprise
        surprise = eps_actual - eps_estimate
        surprise_pct = self._calculate_surprise_pct(eps_actual, eps_estimate)
        
        # Return as dict
        return {
            'announcement_date': last_date,
            'epsActual': float(eps_actual),
            'epsEstimate': float(eps_estimate),
            'surprise': float(surprise),
            'surprise_pct': surprise_pct,
            'beat_estimates': eps_actual > eps_estimate
        }
    
    def get_earnings_for_date_range(
        self,
        symbol: str,
        start_date: date,
        end_date: date
    ) -> pd.DataFrame:
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Initialize result DataFrame
        result = pd.DataFrame(index=date_range)
        result.index.name = 'Date'
        
        # For each date, get the last announcement
        last_announcement_dates = []
        days_since_announcement = []
        surprise_pcts = []
        beat_estimates = []
        
        for current_date in date_range:
            earnings = self.get_last_announcement_before(symbol, current_date.date())
            
            if earnings is not None:
                last_announcement_dates.append(earnings['announcement_date'])
                days_since = (current_date.date() - earnings['announcement_date'].date()).days
                days_since_announcement.append(days_since)
                surprise_pcts.append(earnings['surprise_pct'])
                beat_estimates.append(1 if earnings['beat_estimates'] else 0)
            else:
                last_announcement_dates.append(None)
                days_since_announcement.append(999)  # No earnings yet
                surprise_pcts.append(0)
                beat_estimates.append(0)
        
        result['last_announcement_date'] = last_announcement_dates
        result['days_since_announcement'] = days_since_announcement
        result['last_earnings_surprise_%'] = surprise_pcts
        result['beat_estimates'] = beat_estimates
        
        return result
    
    def _get_earnings_history(self, symbol: str) -> Optional[pd.DataFrame]:

        if symbol.startswith('^'):
            return None
        
        # Check memory cache
        if symbol in self.earnings_cache:
            return self.earnings_cache[symbol]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{symbol.replace('.NS', '')}_earnings.parquet"
        
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self.earnings_cache[symbol] = df
                return df
            except Exception as e:
                print(f"   ⚠️ Cache read failed for {symbol}: {e}")
        
        # Fetch from yfinance
        print(f"   📥 Fetching earnings for {symbol} (using earnings_dates)...")
        
        try:
            ticker = yf.Ticker(symbol)
            
            # ✅ FIX: Use earnings_dates instead of earnings_history!
            earnings = ticker.earnings_dates
            
            if earnings is None or earnings.empty:
                print(f"   ⚠️ No earnings data for {symbol}")
                return None
            
            # Filter out future earnings (NaN in Reported EPS)
            earnings = earnings[~earnings['Reported EPS'].isna()].copy()
            
            if earnings.empty:
                print(f"   ⚠️ No reported earnings for {symbol}")
                return None
            
            # Sort by date (oldest first)
            earnings = earnings.sort_index()
            
            print(f"   ✅ Found {len(earnings)} reported earnings")
            print(f"      Earliest: {earnings.index[0].date()}")
            print(f"      Latest: {earnings.index[-1].date()}")
            
            # Save to cache
            try:
                earnings.to_parquet(cache_file)
                print(f"   ✅ Cached to disk")
            except Exception as e:
                print(f"   ⚠️ Cache write failed: {e}")
            
            # Save to memory
            self.earnings_cache[symbol] = earnings
            
            return earnings
            
        except Exception as e:
            print(f"   ❌ Failed to fetch earnings for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _calculate_surprise_pct(self, actual: float, estimate: float) -> float:
        
        if estimate == 0 or pd.isna(estimate):
            return 0.0
        
        surprise = actual - estimate
        surprise_pct = (surprise / abs(estimate)) * 100
        
        return round(surprise_pct, 2)
    
    def clear_cache(self, symbol: Optional[str] = None):
        
        if symbol:
            cache_file = self.cache_dir / f"{symbol.replace('.NS', '')}_earnings.parquet"
            
            if cache_file.exists():
                cache_file.unlink()
            
            if symbol in self.earnings_cache:
                del self.earnings_cache[symbol]
            
            print(f"🗑️ Cleared earnings cache for {symbol}")
        else:
            for file in self.cache_dir.glob("*_earnings.parquet"):
                file.unlink()
            
            self.earnings_cache.clear()
            
            print(f"🗑️ Cleared all earnings cache")
    
    def refresh_symbol(self, symbol: str):
        
        self.clear_cache(symbol)
        self._get_earnings_history(symbol)

        print(f"🔄 Refreshed earnings data for {symbol}")


def update_universe(max_fetches: int = 150, stale_days: int = 100):
    """
    Refresh earnings caches for every symbol in data/stocks. Used by the
    daily GitHub Actions run so the PEAD alpha has data IN THE CLOUD
    (data/ is gitignored, so without this the cloud never sees earnings).

    A symbol is refetched only if it has no cache or its latest reported
    earnings is older than `stale_days` (results are quarterly, so ~100 days
    means a new announcement is due/out). Fetches are capped per run to
    stay friendly with yfinance rate limits — the daily cadence catches up
    within a couple of runs.
    """
    import time

    stocks_dir = Path("data/stocks")
    skip = {'NIFTY50', 'NIFTYBANK', 'INDIAVIX'}
    symbols = sorted(p.stem for p in stocks_dir.glob("*.parquet")
                     if p.stem not in skip)
    if not symbols:
        print("No price parquets found — nothing to update")
        return

    cal = EarningsCalendar()
    today = date.today()
    fetched = fresh = failed = 0

    for sym in symbols:
        if fetched >= max_fetches:
            print(f"   Fetch cap ({max_fetches}) reached — "
                  "remaining symbols roll to the next run")
            break
        cache_file = cal.cache_dir / f"{sym}_earnings.parquet"
        if cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                last = pd.to_datetime(df.index.max()).date()
                if (today - last).days < stale_days:
                    fresh += 1
                    continue
            except Exception:
                pass
            try:
                cache_file.unlink()
            except OSError:
                pass
        if f"{sym}.NS" in cal.earnings_cache:
            del cal.earnings_cache[f"{sym}.NS"]
        fetched += 1
        if cal._get_earnings_history(f"{sym}.NS") is None:
            failed += 1
        time.sleep(0.5)   # be gentle with yfinance

    print(f"\nEarnings update: {fresh} fresh, {fetched} fetched "
          f"({failed} failed), {len(symbols)} symbols total")


if __name__ == "__main__":
    import sys as _sys
    if '--update' in _sys.argv:
        update_universe()
        _sys.exit(0)

    print("="*70)
    print("EARNINGS CALENDAR TEST (FIXED VERSION)")
    print("="*70)
    
    calendar = EarningsCalendar()
    
    # Test with RELIANCE
    symbol = "RELIANCE.NS"
    
    print(f"\n📊 Testing with {symbol}")
    print("-" * 70)
    
    # Test 1: Recent date (should find earnings!)
    test_date = date(2024, 12, 1)
    
    print(f"\nTest 1: Last earnings before {test_date}")
    earnings = calendar.get_last_announcement_before(symbol, test_date)
    
    if earnings:
        print(f"   ✅ Found earnings:")
        print(f"      Announcement Date: {earnings['announcement_date'].date()}")
        print(f"      Actual EPS: ₹{earnings['epsActual']:.2f}")
        print(f"      Estimate EPS: ₹{earnings['epsEstimate']:.2f}")
        print(f"      Surprise: {earnings['surprise_pct']:+.2f}%")
        print(f"      Beat Estimates: {'Yes' if earnings['beat_estimates'] else 'No'}")
    else:
        print(f"   ⚠️ No earnings before {test_date}")
    
    # Test 2: Get timeline
    print(f"\nTest 2: Earnings timeline Oct-Dec 2024")
    
    df = calendar.get_earnings_for_date_range(
        symbol,
        date(2024, 10, 1),
        date(2024, 12, 31)
    )
    
    print(f"   ✅ Generated {len(df)} days of point-in-time data")
    
    # Show when earnings updated
    changes = df[df['last_announcement_date'].shift() != df['last_announcement_date']]
    
    if not changes.empty:
        print(f"\n   📅 Earnings announcements in this period:")
        for idx in changes.index:
            ann_date = df.loc[idx, 'last_announcement_date']
            surprise = df.loc[idx, 'last_earnings_surprise_%']
            beat = df.loc[idx, 'beat_estimates']
            print(f"      {ann_date.date()}: {surprise:+.2f}% {'✅ Beat' if beat else '❌ Miss'}")
    
    print(f"\n   Sample data:")
    print(df[['days_since_announcement', 'last_earnings_surprise_%', 'beat_estimates']].tail(10))
    
    print("\n" + "="*70)
    print("✅ FIXED VERSION TEST COMPLETE!")
    print("="*70)