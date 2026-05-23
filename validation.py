import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ValidationResult:

    is_valid: bool
    score: float
    errors: List[str]
    warnings: List[str]
    
    def __str__(self):
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        return f"{status} | Score: {self.score:.0f}/100 | Errors: {len(self.errors)} | Warnings: {len(self.warnings)}"


class DataValidator:
    
    def __init__(self, min_score: float = 80.0):

        self.min_score = min_score
    
    def validate_price_data(self, df: pd.DataFrame, symbol: str) -> ValidationResult:

        errors = []
        warnings = []
        score = 100.0
        
        # CHECK 1: Enough data?
        if len(df) < 100:
            errors.append(f"Insufficient data: {len(df)} rows (need 100+)")
            score -= 50
        elif len(df) < 200:
            warnings.append(f"Limited data: {len(df)} rows (200+ recommended)")
            score -= 10
        
        # CHECK 2: Required columns exist?
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            errors.append(f"Missing columns: {missing}")
            score -= 40
            return ValidationResult(False, score, errors, warnings)
        
        # CHECK 3: Missing values?
        total_cells = len(df) * len(required_cols)
        missing_cells = df[required_cols].isnull().sum().sum()
        missing_pct = missing_cells / total_cells
        
        if missing_pct > 0.10:
            errors.append(f"Too many missing values: {missing_pct:.1%}")
            score -= 30
        elif missing_pct > 0.05:
            warnings.append(f"Some missing values: {missing_pct:.1%}")
            score -= 10
        
        # CHECK 4: Prices are positive?
        if (df['Close'] <= 0).any():
            errors.append("Non-positive prices detected!")
            score -= 40
        
        # CHECK 5: High >= Low?
        if (df['High'] < df['Low']).any():
            errors.append("High < Low detected (data corruption!)")
            score -= 40
        
        # CHECK 6: Close between High and Low?
        outside_range = (df['Close'] > df['High']) | (df['Close'] < df['Low'])
        if outside_range.any():
            errors.append("Close outside High/Low range (data corruption!)")
            score -= 30
        
        # CHECK 7: Crazy price jumps?
        returns = df['Close'].pct_change()
        extreme_moves = (abs(returns) > 0.20).sum()
        
        if extreme_moves > len(df) * 0.05:
            warnings.append(f"Many extreme moves: {extreme_moves} days >20%")
            score -= 15
        elif extreme_moves > 0:
            warnings.append(f"Some extreme moves: {extreme_moves} days >20%")
            score -= 5
        
        # CHECK 8: Volume sanity?
        zero_volume_days = (df['Volume'] == 0).sum()
        
        if zero_volume_days > len(df) * 0.10:
            warnings.append(f"Many zero-volume days: {zero_volume_days}")
            score -= 15
        elif zero_volume_days > 0:
            warnings.append(f"Some zero-volume days: {zero_volume_days}")
            score -= 5
        
        # CHECK 9: Data is recent?
        if isinstance(df.index, pd.DatetimeIndex):
            last_date = df.index[-1]
            days_old = (datetime.now() - last_date).days
            
            if days_old > 7:
                warnings.append(f"Data is {days_old} days old")
                score -= min(10, days_old)
        
        # Final decision
        is_valid = len(errors) == 0 and score >= self.min_score
        
        return ValidationResult(
            is_valid=is_valid,
            score=max(0, score),
            errors=errors,
            warnings=warnings
        )
    
    def validate_features(self, features: pd.DataFrame, symbol: str) -> ValidationResult:

        errors = []
        warnings = []
        score = 100.0
        
        # CHECK 1: Enough features?
        if len(features.columns) < 10:
            errors.append(f"Too few features: {len(features.columns)}")
            score -= 40
        
        # CHECK 2: NaN values?
        nan_cols = features.columns[features.isnull().any()].tolist()
        
        if len(nan_cols) > len(features.columns) * 0.10:
            errors.append(f"Too many NaN columns: {len(nan_cols)}/{len(features.columns)}")
            score -= 30
        elif nan_cols:
            warnings.append(f"Some NaN columns: {len(nan_cols)}")
            score -= 10
        
        # CHECK 3: Infinite values?
        inf_cols = features.columns[np.isinf(features).any()].tolist()
        
        if inf_cols:
            errors.append(f"Infinite values in: {inf_cols[:5]}")
            score -= 30
        
        # CHECK 4: Constant features (no variation)?
        constant_cols = features.columns[features.nunique() == 1].tolist()
        
        if len(constant_cols) > len(features.columns) * 0.20:
            warnings.append(f"Many constant features: {len(constant_cols)}")
            score -= 15
        
        # Final decision
        is_valid = len(errors) == 0 and score >= self.min_score
        
        return ValidationResult(
            is_valid=is_valid,
            score=max(0, score),
            errors=errors,
            warnings=warnings
        )


# Test function - run this to verify it works
if __name__ == "__main__":
    print("="*70)
    print("DATA VALIDATOR TEST")
    print("="*70)
    
    # Test 1: Good data
    print("\n1. Testing GOOD data:")
    dates = pd.date_range('2024-01-01', periods=200, freq='D')
    good_data = pd.DataFrame({
        'Open': np.random.uniform(100, 110, 200),
        'High': np.random.uniform(110, 120, 200),
        'Low': np.random.uniform(90, 100, 200),
        'Close': np.random.uniform(100, 110, 200),
        'Volume': np.random.uniform(1000000, 2000000, 200)
    }, index=dates)
    
    validator = DataValidator()
    result = validator.validate_price_data(good_data, "TEST_GOOD")
    print(f"   Result: {result}")
    
    # Test 2: Bad data
    print("\n2. Testing BAD data:")
    bad_data = pd.DataFrame({
        'Open': [100, -10, 110],  # Negative price!
        'High': [110, 105, 120],
        'Low': [90, 95, 100],
        'Close': [105, 100, 115],
        'Volume': [1000000, 0, 2000000]
    })
    
    result = validator.validate_price_data(bad_data, "TEST_BAD")
    print(f"   Result: {result}")
    if result.errors:
        print(f"   Errors: {result.errors}")
    if result.warnings:
        print(f"   Warnings: {result.warnings}")
    
    print("\n" + "="*70)
    print("✅ If you see results above, validation module works!")
    print("="*70)