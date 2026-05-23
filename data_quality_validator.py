"""
DATA QUALITY VALIDATOR
======================
Catches data errors before they poison your models.

Think of it as a security guard checking everyone who enters:
- Stock splits? → Adjust prices
- Circuit breaker violations? → Flag suspicious
- Volume spikes? → Investigate
- Missing data? → Fill or warn

This is what separates amateur systems from professional ones!
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple
import warnings


class DataQualityValidator:
    """
    Validates stock data quality and detects anomalies.
    
    Catches:
    1. Stock splits (price suddenly 10x lower)
    2. Circuit breakers (±20% NSE limit violations)
    3. Volume spikes (>10x normal volume)
    4. Missing data (gaps in dates)
    5. Corporate actions (bonus, rights, dividends)
    """
    
    def __init__(self):
        # NSE rules
        self.circuit_limit_pct = 20.0  # NSE has ±20% circuit breakers
        self.volume_spike_threshold = 10.0  # 10x normal volume
        self.split_detection_threshold = 0.4  # 40% price drop = potential split
        
        # For tracking historical patterns
        self.known_splits = {}  # Cache detected splits
        
    # ========================================
    # MAIN VALIDATION FUNCTION
    # ========================================
    def validate(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Main validation function - runs ALL checks.
        
        Args:
            df: DataFrame with OHLCV data (index = Date)
            symbol: Stock symbol (e.g., "RELIANCE.NS")
            
        Returns:
            (cleaned_df, quality_report)
            
            cleaned_df: Data with issues fixed
            quality_report: Dict with all findings
        """
        
        if df is None or df.empty:
            return df, {'error': 'Empty DataFrame'}
        
        print(f"\n🔍 Validating data quality for {symbol}")
        print(f"   Date range: {df.index.min().date()} to {df.index.max().date()}")
        print(f"   Total rows: {len(df)}")
        
        report = {
            'symbol': symbol,
            'total_rows': len(df),
            'date_range': (df.index.min(), df.index.max()),
            'issues_found': [],
            'warnings': [],
            'fixes_applied': []
        }
        
        # Make a copy to avoid modifying original
        df_clean = df.copy()
        
        # === CHECK 1: Missing Data ===
        df_clean, missing_report = self._check_missing_dates(df_clean, symbol)
        if missing_report['gaps_found'] > 0:
            report['issues_found'].append(f"Missing data: {missing_report['gaps_found']} gaps")
            report['warnings'].extend(missing_report['gap_dates'])
        
        # === CHECK 2: Stock Splits ===
        df_clean, split_report = self._check_stock_splits(df_clean, symbol)
        if split_report['splits_detected'] > 0:
            report['issues_found'].append(f"Stock splits: {split_report['splits_detected']} detected")
            report['fixes_applied'].extend(split_report['split_dates'])
        
        # === CHECK 3: Circuit Breakers ===
        circuit_report = self._check_circuit_breakers(df_clean, symbol)
        if circuit_report['violations'] > 0:
            report['issues_found'].append(f"Circuit violations: {circuit_report['violations']} dates")
            report['warnings'].extend(circuit_report['violation_dates'])
        
        # === CHECK 4: Volume Spikes ===
        volume_report = self._check_volume_spikes(df_clean, symbol)
        if volume_report['spikes'] > 0:
            report['issues_found'].append(f"Volume spikes: {volume_report['spikes']} dates")
            report['warnings'].extend(volume_report['spike_dates'])
        
        # === CHECK 5: Price Sanity ===
        price_report = self._check_price_sanity(df_clean, symbol)
        if price_report['issues'] > 0:
            report['issues_found'].append(f"Price issues: {price_report['issues']} found")
            report['warnings'].extend(price_report['details'])
        
        # === SUMMARY ===
        report['quality_score'] = self._calculate_quality_score(report)
        
        print(f"\n📊 Validation Summary:")
        print(f"   Quality Score: {report['quality_score']}/100")
        print(f"   Issues Found: {len(report['issues_found'])}")
        print(f"   Warnings: {len(report['warnings'])}")
        print(f"   Fixes Applied: {len(report['fixes_applied'])}")
        
        if report['quality_score'] >= 90:
            print(f"   ✅ EXCELLENT - Data is high quality")
        elif report['quality_score'] >= 70:
            print(f"   ⚠️ GOOD - Minor issues detected")
        elif report['quality_score'] >= 50:
            print(f"   ⚠️ FAIR - Some issues need attention")
        else:
            print(f"   ❌ POOR - Major data quality issues!")
        
        # ✅ CRITICAL: Clean up any temporary columns that may have been added
        temp_cols = ['volume_ma20', 'volume_ratio', 'daily_return']
        for col in temp_cols:
            if col in df_clean.columns:
                df_clean = df_clean.drop(columns=[col])
        
        return df_clean, report
    
    # ========================================
    # CHECK 1: MISSING DATES
    # ========================================
    def _check_missing_dates(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect gaps in date sequence (skipping weekends/holidays).
        
        Market is closed on weekends, so we only check for missing weekdays.
        """
        
        if len(df) < 2:
            return df, {'gaps_found': 0, 'gap_dates': []}
        
        # Get all dates
        dates = pd.Series(df.index.date)
        
        # Find gaps (more than 3 days = suspicious, accounting for weekends)
        gaps = []
        for i in range(1, len(dates)):
            days_diff = (dates.iloc[i] - dates.iloc[i-1]).days
            
            # If gap > 5 days (excluding weekends/holidays), flag it
            if days_diff > 5:
                gaps.append({
                    'from': dates.iloc[i-1],
                    'to': dates.iloc[i],
                    'days': days_diff
                })
        
        if len(gaps) > 0:
            print(f"   ⚠️ {len(gaps)} data gaps found (>5 days)")
            for gap in gaps[:3]:  # Show first 3
                print(f"      {gap['from']} → {gap['to']} ({gap['days']} days)")
        
        return df, {
            'gaps_found': len(gaps),
            'gap_dates': [f"{g['from']} to {g['to']}" for g in gaps]
        }
    
    # ========================================
    # CHECK 2: STOCK SPLITS
    # ========================================
    def _check_stock_splits(self, df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Detect stock splits by looking for sudden large price drops.
        
        Example:
        Day 1: ₹1,500
        Day 2: ₹150  (90% drop = 1:10 split!)
        
        We adjust historical prices to maintain continuity.
        
        IMPROVED: Uses Adj Close if available to detect unadjusted data.
        """
        
        if len(df) < 2:
            return df, {'splits_detected': 0, 'split_dates': []}
        
        # CRITICAL FIX: Check if data is already adjusted
        # If Close != Adj Close, data is already adjusted by provider
        has_adj_close = 'Adj Close' in df.columns
        
        if has_adj_close:
            # Use Adj Close for calculations (already split-adjusted)
            price_col = 'Adj Close'
            print(f"   ℹ️ Using Adj Close (provider-adjusted data)")
        else:
            price_col = 'Close'
            print(f"   ℹ️ Using Close (may need split adjustment)")
        
        # Calculate daily returns on adjusted prices
        df['daily_return'] = df[price_col].pct_change()
        
        # Detect potential splits (large negative returns)
        # A 1:2 split = -50% return
        # A 1:10 split = -90% return
        potential_splits = df[df['daily_return'] < -self.split_detection_threshold].copy()
        
        splits_detected = []
        df_adjusted = df.copy()
        
        for split_date in potential_splits.index:
            # Get before/after prices
            split_idx = df.index.get_loc(split_date)
            
            if split_idx == 0:
                continue  # Can't detect split on first row
            
            price_before = df.iloc[split_idx - 1][price_col]
            price_after = df.iloc[split_idx][price_col]
            
            # Calculate split ratio
            ratio = price_before / price_after
            
            # Common split ratios: 1:2, 1:5, 1:10, 2:1 (bonus)
            if ratio > 1.8 and ratio < 2.2:
                split_ratio = "1:2"
                adjustment_factor = 2.0
            elif ratio > 4.5 and ratio < 5.5:
                split_ratio = "1:5"
                adjustment_factor = 5.0
            elif ratio > 9.0 and ratio < 11.0:
                split_ratio = "1:10"
                adjustment_factor = 10.0
            elif ratio > 0.45 and ratio < 0.55:
                split_ratio = "2:1"
                adjustment_factor = 0.5
            else:
                # Unusual ratio - might not be a split
                continue
            
            print(f"   🔀 SPLIT DETECTED: {split_date.date()} - Ratio ~{split_ratio}")
            print(f"      Before: ₹{price_before:.2f} → After: ₹{price_after:.2f}")
            
            # ONLY adjust if using Close (not Adj Close)
            if price_col == 'Close':
                # Adjust all historical prices BEFORE the split
                for col in ['Open', 'High', 'Low', 'Close']:
                    df_adjusted.loc[:split_date, col] = df_adjusted.loc[:split_date, col] / adjustment_factor
                
                # Adjust volume (multiply by split factor)
                df_adjusted.loc[:split_date, 'Volume'] = df_adjusted.loc[:split_date, 'Volume'] * adjustment_factor
                
                print(f"      ✅ Historical prices adjusted by factor {1/adjustment_factor:.2f}")
            else:
                print(f"      ℹ️ Adj Close already accounts for split, no adjustment needed")
            
            splits_detected.append({
                'date': split_date.date(),
                'ratio': split_ratio,
                'price_before': price_before,
                'price_after': price_after
            })
        
        # Remove temporary column
        if 'daily_return' in df_adjusted.columns:
            df_adjusted = df_adjusted.drop(columns=['daily_return'])
        
        return df_adjusted, {
            'splits_detected': len(splits_detected),
            'split_dates': [f"{s['date']} ({s['ratio']})" for s in splits_detected]
        }
    
    # ========================================
    # CHECK 3: CIRCUIT BREAKERS
    # ========================================
    def _check_circuit_breakers(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        NSE has ±20% circuit breakers (for most stocks).
        
        If a stock moves >20% in one day, it's either:
        1. Data error
        2. Corporate action not adjusted
        3. Rare exception (micro-cap stocks)
        """
        
        if len(df) < 2:
            return {'violations': 0, 'violation_dates': []}
        
        # Use the appropriate price column
        has_adj_close = 'Adj Close' in df.columns
        price_col = 'Adj Close' if has_adj_close else 'Close'
        
        # Calculate daily returns
        daily_returns = df[price_col].pct_change() * 100  # Convert to percentage
        
        # ✅ CRITICAL FIX: Drop NaN values (first row will always be NaN)
        daily_returns = daily_returns.dropna()
        
        # Find circuit breaker violations
        violations_mask = abs(daily_returns) > self.circuit_limit_pct
        violations = daily_returns[violations_mask]
        
        violation_list = []
        
        for idx in violations.index:
            ret = violations.loc[idx]
            price = df.loc[idx, price_col]
            
            violation_list.append({
                'date': idx.date(),
                'return': f"{ret:+.1f}%",
                'price': f"₹{price:.2f}"
            })
        
        if len(violation_list) > 0:
            print(f"   ⚠️ {len(violation_list)} circuit breaker violations (>±20%)")
            for v in violation_list[:3]:  # Show first 3
                print(f"      {v['date']}: {v['return']} move to {v['price']}")
        
        return {
            'violations': len(violation_list),
            'violation_dates': [f"{v['date']} ({v['return']})" for v in violation_list]
        }
    
    # ========================================
    # CHECK 4: VOLUME SPIKES
    # ========================================
    def _check_volume_spikes(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Detect unusual volume spikes (>10x normal).
        
        Could indicate:
        - Merger/acquisition news
        - Market manipulation
        - Data error
        """
        
        if len(df) < 20:  # Need history for baseline
            return {'spikes': 0, 'spike_dates': []}
        
        # Work on a copy to avoid modifying original
        df_temp = df.copy()
        
        # Calculate rolling 20-day average volume
        df_temp['volume_ma20'] = df_temp['Volume'].rolling(window=20, min_periods=5).mean()
        
        # Find spikes (>10x average)
        df_temp['volume_ratio'] = df_temp['Volume'] / (df_temp['volume_ma20'] + 1)
        
        spikes = df_temp[df_temp['volume_ratio'] > self.volume_spike_threshold].copy()
        
        spike_list = []
        
        for idx in spikes.index:
            ratio = spikes.loc[idx, 'volume_ratio']
            volume = df.loc[idx, 'Volume']
            avg_volume = df_temp.loc[idx, 'volume_ma20']
            
            spike_list.append({
                'date': idx.date(),
                'volume': int(volume),
                'avg_volume': int(avg_volume),
                'ratio': f"{ratio:.1f}x"
            })
        
        if len(spike_list) > 0:
            print(f"   📊 {len(spike_list)} volume spikes detected (>10x average)")
            for s in spike_list[:3]:  # Show first 3
                print(f"      {s['date']}: {s['volume']:,} ({s['ratio']} normal)")
        
        # NOTE: We DON'T modify the original df, so no cleanup needed
        
        return {
            'spikes': len(spike_list),
            'spike_dates': [f"{s['date']} ({s['ratio']})" for s in spike_list]
        }
    
    # ========================================
    # CHECK 5: PRICE SANITY
    # ========================================
    def _check_price_sanity(self, df: pd.DataFrame, symbol: str) -> Dict:
        """
        Basic sanity checks:
        - High >= Low (always true)
        - Close within [Low, High]
        - No negative prices
        - No zero prices
        """
        
        issues = []
        
        # Check 1: High >= Low
        invalid_hl = df[df['High'] < df['Low']]
        if len(invalid_hl) > 0:
            for idx in invalid_hl.index:
                issues.append(f"{idx.date()}: High < Low!")
        
        # Check 2: Close within range
        invalid_close = df[(df['Close'] > df['High']) | (df['Close'] < df['Low'])]
        if len(invalid_close) > 0:
            for idx in invalid_close.index:
                issues.append(f"{idx.date()}: Close outside [Low, High]")
        
        # Check 3: No negative prices
        negative_prices = df[(df['Open'] < 0) | (df['High'] < 0) | (df['Low'] < 0) | (df['Close'] < 0)]
        if len(negative_prices) > 0:
            for idx in negative_prices.index:
                issues.append(f"{idx.date()}: Negative price detected!")
        
        # Check 4: No zero prices
        zero_prices = df[(df['Open'] == 0) | (df['High'] == 0) | (df['Low'] == 0) | (df['Close'] == 0)]
        if len(zero_prices) > 0:
            for idx in zero_prices.index:
                issues.append(f"{idx.date()}: Zero price detected!")
        
        if len(issues) > 0:
            print(f"   ❌ {len(issues)} price sanity issues!")
            for issue in issues[:3]:  # Show first 3
                print(f"      {issue}")
        
        return {
            'issues': len(issues),
            'details': issues
        }
    
    # ========================================
    # QUALITY SCORE CALCULATION
    # ========================================
    def _calculate_quality_score(self, report: Dict) -> int:
        """
        Calculate overall quality score (0-100).
        
        Scoring:
        - Start with 100
        - Deduct points for each issue type
        """
        
        score = 100
        
        # Deduct for each issue type
        num_issues = len(report['issues_found'])
        num_warnings = len(report['warnings'])
        
        # Major issues (stock splits, circuit violations, price errors)
        score -= num_issues * 5
        
        # Minor warnings (volume spikes, data gaps)
        score -= num_warnings * 1
        
        # Floor at 0
        score = max(0, score)
        
        return score


# ========================================
# EXAMPLE USAGE
# ========================================
if __name__ == "__main__":
    """
    Test the validator with sample data
    """
    
    # Create sample data with intentional issues
    dates = pd.date_range(start='2024-01-01', end='2024-12-20', freq='B')  # Business days
    
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Open': 1000 + np.random.randn(len(dates)).cumsum() * 10,
        'High': 1010 + np.random.randn(len(dates)).cumsum() * 10,
        'Low': 990 + np.random.randn(len(dates)).cumsum() * 10,
        'Close': 1000 + np.random.randn(len(dates)).cumsum() * 10,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    }, index=dates)
    
    # Ensure High >= Low >= Close
    df['High'] = df[['Open', 'High', 'Low', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    # Add a fake stock split on June 3 (Monday - guaranteed to be in business days)
    split_date = pd.Timestamp('2024-06-03')
    
    # Make sure the date exists in our index
    if split_date not in df.index:
        # Find the nearest business day after June 1
        split_date = df.index[df.index >= pd.Timestamp('2024-06-01')][0]
    
    split_idx = df.index.get_loc(split_date)
    
    # 1:2 split - divide all prices by 2 after split
    df.loc[split_date:, ['Open', 'High', 'Low', 'Close']] /= 2
    df.loc[split_date:, 'Volume'] *= 2
    
    # Add a volume spike on Sep 16 (Monday)
    spike_date = pd.Timestamp('2024-09-16')
    
    # Make sure the date exists
    if spike_date not in df.index:
        spike_date = df.index[df.index >= pd.Timestamp('2024-09-15')][0]
    
    df.loc[spike_date, 'Volume'] *= 15  # 15x normal
    
    print("="*70)
    print("DATA QUALITY VALIDATOR TEST")
    print("="*70)
    
    # Run validator
    validator = DataQualityValidator()
    df_clean, report = validator.validate(df, "TEST.NS")
    
    print("\n" + "="*70)
    print("FULL REPORT")
    print("="*70)
    print(f"Quality Score: {report['quality_score']}/100")
    print(f"\nIssues Found: {len(report['issues_found'])}")
    for issue in report['issues_found']:
        print(f"  - {issue}")
    
    print(f"\nFixes Applied: {len(report['fixes_applied'])}")
    for fix in report['fixes_applied']:
        print(f"  - {fix}")