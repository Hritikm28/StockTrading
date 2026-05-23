"""
DATA CONTRACTS - Aladdin-style data validation
==============================================
Ensures data integrity at every component boundary.
NO SILENT FAILURES - fail fast with clear errors.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime


@dataclass
class DataQualityReport:
    """Report on data quality for a single stock"""
    symbol: str
    is_valid: bool
    score: float  # 0-100
    missing_columns: List[str]
    stale_data: bool
    data_completeness: float  # 0-1
    issues: List[str]
    warnings: List[str]
    
    def to_dict(self) -> dict:
        return {
            'symbol': self.symbol,
            'is_valid': self.is_valid,
            'score': self.score,
            'missing_columns': self.missing_columns,
            'stale_data': self.stale_data,
            'data_completeness': self.data_completeness,
            'issues': self.issues,
            'warnings': self.warnings
        }


class DataContracts:
    """
    Central data contract definitions.
    
    These are the columns REQUIRED for the system to function.
    If any are missing, the system should FAIL FAST.
    """
    
    # OHLCV - Absolutely required for any analysis
    OHLCV_REQUIRED = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Price columns that can be computed
    PRICE_COMPUTED = ['Adj Close', 'Returns']
    
    # Minimum features for ML predictions
    ML_MINIMUM_FEATURES = [
        'Returns', 'RSI_14', 'MACD', 'SMA_50', 'SMA_200',
        'BB_Upper_20', 'BB_Lower_20', 'ATR_14', 'OBV', 'Volume_Ratio'
    ]
    
    # External data features (optional but tracked)
    EXTERNAL_FEATURES = [
        'FII_Net', 'DII_Net', 'PCR', 'News_Sentiment',
        'Sector_Momentum', 'Market_Breadth'
    ]
    
    # Model metadata keys (NOT model objects)
    MODEL_METADATA_KEYS = {
        'feature_cols', 'scaler', 'learned_weights', 'training_date', 'version',
        'moe', 'tabnet_meta', 'attention_ensemble', 'attention_model_names',
        'bma_weights', 'posterior_samples', 'conformal_threshold',
        'adversarial_auc', 'nas_config', 'diverse_selection', 'flaml',
        'external_features', 'external_quality', 'external_importance',
        'data_quality_score', 'data_completeness'
    }
    
    # Actual model keys
    BASE_MODEL_KEYS = {'xgb', 'lgb', 'cat', 'rf', 'et'}
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame, symbol: str = "Unknown") -> DataQualityReport:
        """Validate that OHLCV data is present and valid"""
        issues = []
        warnings = []
        missing = []
        
        # Check required columns
        for col in DataContracts.OHLCV_REQUIRED:
            if col not in df.columns:
                missing.append(col)
                issues.append(f"Missing required column: {col}")
        
        if missing:
            return DataQualityReport(
                symbol=symbol,
                is_valid=False,
                score=0.0,
                missing_columns=missing,
                stale_data=False,
                data_completeness=0.0,
                issues=issues,
                warnings=warnings
            )
        
        # Check for NaN values
        nan_counts = df[DataContracts.OHLCV_REQUIRED].isna().sum()
        for col, count in nan_counts.items():
            if count > 0:
                pct = count / len(df) * 100
                if pct > 5:
                    issues.append(f"{col} has {pct:.1f}% NaN values")
                else:
                    warnings.append(f"{col} has {count} NaN values ({pct:.1f}%)")
        
        # Check for zero prices (likely bad data)
        if (df['Close'] <= 0).any():
            issues.append("Found zero or negative Close prices")
        
        # Check for zero volume (suspicious)
        zero_vol_pct = (df['Volume'] == 0).sum() / len(df) * 100
        if zero_vol_pct > 10:
            warnings.append(f"{zero_vol_pct:.1f}% of days have zero volume")
        
        # Calculate completeness
        completeness = 1 - df[DataContracts.OHLCV_REQUIRED].isna().mean().mean()
        
        # Calculate score
        score = 100.0
        score -= len(issues) * 20  # Major issues
        score -= len(warnings) * 5  # Minor issues
        score = max(0, score)
        
        return DataQualityReport(
            symbol=symbol,
            is_valid=len(issues) == 0,
            score=score,
            missing_columns=missing,
            stale_data=False,
            data_completeness=completeness,
            issues=issues,
            warnings=warnings
        )
    
    @staticmethod
    def ensure_returns_column(df: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
        """
        CRITICAL: Ensure Returns column exists.
        This is called before ANY feature engineering or ML.
        """
        if 'Returns' in df.columns:
            return df
        
        df = df.copy()
        
        # Try Adj Close first (preferred for dividends/splits)
        if 'Adj Close' in df.columns:
            df['Returns'] = df['Adj Close'].pct_change()
            if symbol:
                print(f"   [DataContract] {symbol}: Created Returns from Adj Close")
        elif 'Close' in df.columns:
            df['Returns'] = df['Close'].pct_change()
            if symbol:
                print(f"   [DataContract] {symbol}: Created Returns from Close")
        else:
            # CRITICAL FAILURE - cannot proceed without price data
            raise ValueError(f"Cannot create Returns: no Close or Adj Close column for {symbol}")
        
        # Shift by 1 to avoid look-ahead bias
        df['Returns'] = df['Returns'].shift(1)
        
        return df
    
    @staticmethod
    def validate_for_ml(df: pd.DataFrame, symbol: str = "Unknown") -> DataQualityReport:
        """Validate that data has minimum features for ML"""
        base_report = DataContracts.validate_ohlcv(df, symbol)
        
        if not base_report.is_valid:
            return base_report
        
        # Check ML features
        missing_ml = []
        for col in DataContracts.ML_MINIMUM_FEATURES:
            if col not in df.columns:
                missing_ml.append(col)
        
        if missing_ml:
            base_report.issues.append(f"Missing ML features: {missing_ml[:5]}...")
            base_report.missing_columns.extend(missing_ml)
            base_report.score -= len(missing_ml) * 2
        
        # Check for sufficient data
        if len(df) < 250:
            base_report.issues.append(f"Insufficient data: {len(df)} rows (need 250+)")
            base_report.is_valid = False
            base_report.score -= 30
        
        base_report.score = max(0, base_report.score)
        base_report.is_valid = len([i for i in base_report.issues if 'Missing required' in i]) == 0
        
        return base_report
    
    @staticmethod
    def validate_external_data(context: dict, symbol: str = "Unknown") -> Tuple[float, List[str]]:
        """
        Validate external data quality and return score + warnings.
        
        Returns:
            Tuple of (quality_score 0-100, list of warnings)
        """
        if not context:
            return 0.0, ["No external data context provided"]
        
        score = 100.0
        warnings = []
        
        # Check each external data source
        checks = {
            'fii_dii': ('FII_Net', 'DII_Net'),
            'options': ('pcr',),
            'news': ('score', 'confidence'),
            'sector': ('momentum',),
            'market_breadth': ('breadth_score',)
        }
        
        for source, required_keys in checks.items():
            if source not in context:
                warnings.append(f"Missing {source} data")
                score -= 15
            else:
                data = context[source]
                for key in required_keys:
                    if key not in data or data.get(key) is None:
                        warnings.append(f"{source}.{key} is missing")
                        score -= 5
                    elif isinstance(data.get(key), float) and np.isnan(data.get(key)):
                        warnings.append(f"{source}.{key} is NaN")
                        score -= 5
        
        return max(0, score), warnings
    
    @staticmethod
    def filter_model_objects(models: dict) -> dict:
        """
        Extract only actual model objects from a models dictionary.
        Use this before iterating over models for predictions.
        """
        return {
            name: model for name, model in models.items()
            if name not in DataContracts.MODEL_METADATA_KEYS
            and hasattr(model, 'predict')
            and hasattr(model, 'fit')
        }
    
    @staticmethod
    def is_model_object(name: str, obj) -> bool:
        """Check if an object is an actual ML model"""
        if name in DataContracts.MODEL_METADATA_KEYS:
            return False
        if not hasattr(obj, 'predict'):
            return False
        if not hasattr(obj, 'fit'):
            return False
        return True


class DataQualityScorecard:
    """
    Tracks data quality across all predictions.
    Every prediction gets a quality score attached.
    """
    
    def __init__(self):
        self.scores = {}
    
    def add_score(self, symbol: str, report: DataQualityReport):
        """Store quality report for a symbol"""
        self.scores[symbol] = report
    
    def get_prediction_confidence_modifier(self, symbol: str) -> float:
        """
        Returns a modifier (0-1) to apply to prediction confidence
        based on data quality.
        
        Example: 
            - 100% quality → 1.0 modifier (full confidence)
            - 50% quality → 0.5 modifier (halve confidence)
            - <30% quality → 0.0 modifier (no confidence - skip trade)
        """
        if symbol not in self.scores:
            return 0.5  # Unknown quality → conservative
        
        report = self.scores[symbol]
        
        if not report.is_valid:
            return 0.0  # Invalid data → no confidence
        
        # Linear scaling from score
        return report.score / 100.0
    
    def should_skip_prediction(self, symbol: str, min_quality: float = 50.0) -> Tuple[bool, str]:
        """
        Determines if a prediction should be skipped due to low quality.
        
        Returns:
            Tuple of (should_skip, reason)
        """
        if symbol not in self.scores:
            return True, "No quality score available"
        
        report = self.scores[symbol]
        
        if not report.is_valid:
            return True, f"Data validation failed: {report.issues[0] if report.issues else 'unknown'}"
        
        if report.score < min_quality:
            return True, f"Quality score {report.score:.0f} below threshold {min_quality}"
        
        if report.stale_data:
            return True, "Data is stale"
        
        return False, "OK"
    
    def get_summary(self) -> dict:
        """Get summary statistics"""
        if not self.scores:
            return {'total': 0, 'valid': 0, 'avg_score': 0}
        
        valid_count = sum(1 for r in self.scores.values() if r.is_valid)
        avg_score = np.mean([r.score for r in self.scores.values()])
        
        return {
            'total': len(self.scores),
            'valid': valid_count,
            'invalid': len(self.scores) - valid_count,
            'avg_score': avg_score,
            'min_score': min(r.score for r in self.scores.values()),
            'max_score': max(r.score for r in self.scores.values())
        }


# Global scorecard instance
_global_scorecard = DataQualityScorecard()


def get_global_scorecard() -> DataQualityScorecard:
    """Get the global data quality scorecard"""
    return _global_scorecard


def validate_and_prepare_data(df: pd.DataFrame, symbol: str) -> Tuple[pd.DataFrame, DataQualityReport]:
    """
    Main entry point for data validation.
    Call this BEFORE any analysis.
    
    Returns:
        Tuple of (prepared_df, quality_report)
    
    Raises:
        ValueError if data is critically invalid
    """
    # Step 1: Basic OHLCV validation
    report = DataContracts.validate_ohlcv(df, symbol)
    
    if not report.is_valid:
        raise ValueError(f"[{symbol}] Critical data failure: {report.issues}")
    
    # Step 2: Ensure Returns exists
    df = DataContracts.ensure_returns_column(df, symbol)
    
    # Step 3: Full ML validation
    report = DataContracts.validate_for_ml(df, symbol)
    
    # Step 4: Store in global scorecard
    _global_scorecard.add_score(symbol, report)
    
    return df, report


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("DATA CONTRACTS TEST")
    print("=" * 70)
    
    # Create test data
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    df = pd.DataFrame({
        'Open': np.random.uniform(100, 200, 300),
        'High': np.random.uniform(100, 200, 300),
        'Low': np.random.uniform(100, 200, 300),
        'Close': np.random.uniform(100, 200, 300),
        'Volume': np.random.randint(1000000, 10000000, 300)
    }, index=dates)
    
    print("\n1. Testing OHLCV validation...")
    report = DataContracts.validate_ohlcv(df, "TEST.NS")
    print(f"   Valid: {report.is_valid}")
    print(f"   Score: {report.score}")
    
    print("\n2. Testing Returns creation...")
    df_with_returns = DataContracts.ensure_returns_column(df, "TEST.NS")
    print(f"   Returns column exists: {'Returns' in df_with_returns.columns}")
    print(f"   Returns sample: {df_with_returns['Returns'].dropna().head(3).tolist()}")
    
    print("\n3. Testing ML validation...")
    report = DataContracts.validate_for_ml(df_with_returns, "TEST.NS")
    print(f"   Valid: {report.is_valid}")
    print(f"   Score: {report.score}")
    print(f"   Missing ML features: {report.missing_columns[:5]}...")
    
    print("\n4. Testing model filtering...")
    fake_models = {
        'xgb': type('FakeModel', (), {'predict': lambda x: x, 'fit': lambda x, y: None})(),
        'feature_cols': ['a', 'b'],
        'scaler': None,
        'external_quality': 0.75
    }
    filtered = DataContracts.filter_model_objects(fake_models)
    print(f"   Original keys: {list(fake_models.keys())}")
    print(f"   Filtered keys: {list(filtered.keys())}")
    
    print("\n" + "=" * 70)
    print("✅ Data Contracts module ready")
    print("=" * 70)

