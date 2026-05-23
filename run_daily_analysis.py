#!/usr/bin/env python
"""
HEADLESS DAILY ANALYSIS RUNNER
==============================
Aladdin-style: Runs independently of UI, generates static reports.

Usage:
    python run_daily_analysis.py                    # Analyze all stocks
    python run_daily_analysis.py --stocks RELIANCE.NS TCS.NS  # Specific stocks
    python run_daily_analysis.py --top 20           # Top 20 by signal strength
    python run_daily_analysis.py --output pdf       # Generate PDF report
    
Schedule this to run at 6:00 AM IST before market opens.
"""

import sys
import io
import os

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

# Import core modules
from data_manager import DataManager
from feature_engine import FeatureEngine
from model_manager import ModelManager
from data_contracts import (
    DataContracts, DataQualityReport, DataQualityScorecard,
    validate_and_prepare_data, get_global_scorecard
)
from config import Config


class DailyAnalysisRunner:
    """
    Headless analysis engine.
    
    Generates predictions without needing Streamlit UI.
    Outputs to JSON, CSV, and optionally PDF.
    """
    
    def __init__(self, output_dir: str = "daily_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.data_manager = DataManager()
        self.feature_engine = FeatureEngine()
        
        # Track results
        self.results = []
        self.errors = []
        self.quality_scorecard = get_global_scorecard()
        
        print("=" * 70)
        print("DAILY ANALYSIS RUNNER - HEADLESS MODE")
        print("=" * 70)
        print(f"Output directory: {self.output_dir}")
        print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    def get_stock_universe(self) -> list:
        """Get list of stocks to analyze"""
        try:
            # Try to load from config
            stocks = []
            
            # Nifty 50
            nifty50 = [
                'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
                'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
                'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
                'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'M&M.NS',
                'BAJFINANCE.NS', 'HCLTECH.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS',
                'JSWSTEEL.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'ONGC.NS',
                'BPCL.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS',
                'TECHM.NS', 'APOLLOHOSP.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'BAJAJFINSV.NS',
                'TATACONSUM.NS', 'BRITANNIA.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'SBILIFE.NS',
                'HDFCLIFE.NS', 'TATAMOTORS.NS', 'UPL.NS', 'LTIM.NS', 'BAJAJ-AUTO.NS'
            ]
            stocks.extend(nifty50)
            
            return stocks
        except Exception as e:
            print(f"Error loading stock universe: {e}")
            return []
    
    def analyze_single_stock(self, symbol: str) -> dict:
        """
        Analyze a single stock and return result.
        
        Returns dict with:
            - symbol
            - signal (BUY/SELL/HOLD)
            - confidence (0-100)
            - data_quality (0-100)
            - prediction_details
            - error (if any)
        """
        result = {
            'symbol': symbol,
            'signal': None,
            'confidence': 0,
            'data_quality': 0,
            'expected_return': 0,
            'risk_reward': 0,
            'model_agreement': 0,
            'timestamp': datetime.now().isoformat(),
            'error': None
        }
        
        try:
            # Step 1: Fetch data
            end_date = date.today()
            start_date = end_date - timedelta(days=365 * 3)  # 3 years
            
            df = self.data_manager.fetch_stock_data_with_features(
                symbol, start_date, end_date, compute_features=True
            )
            
            if df is None or len(df) < 250:
                result['error'] = f"Insufficient data: {len(df) if df is not None else 0} rows"
                return result
            
            # Step 2: Validate with data contracts
            try:
                df, quality_report = validate_and_prepare_data(df, symbol)
                result['data_quality'] = quality_report.score
                
                if not quality_report.is_valid:
                    result['error'] = f"Data quality check failed: {quality_report.issues[0]}"
                    return result
                    
            except ValueError as e:
                result['error'] = str(e)
                return result
            
            # Step 3: Create target for TRAINING only (labels based on future returns)
            # IMPORTANT: This lookahead is INTENTIONAL for label creation only.
            # Features must NOT contain future data — they are computed with .shift(1) in feature_engine.
            horizon = 5  # 5-day prediction
            threshold = 0.02  # 2% move

            # Binary target: 1 = UP (>2%), 0 = DOWN (<-2%)
            future_returns = df['Close'].pct_change(horizon).shift(-horizon)

            # Create target
            df['Target'] = np.where(future_returns > threshold, 1,
                           np.where(future_returns < -threshold, 0, np.nan))

            # Keep ALL feature columns for later prediction on latest rows
            feature_cols = [col for col in df.columns
                          if col not in ['Target', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
                          and not col.startswith('Future')]

            # TRAINING SET: Only rows with a valid target (excludes last `horizon` rows)
            df_clean = df.dropna(subset=['Target'])

            if len(df_clean) < 200:
                result['error'] = f"Insufficient clean data: {len(df_clean)} rows"
                return result

            X_train_full = df_clean[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            y_train_full = df_clean['Target']

            # FIXED: Sort by index to guarantee chronological order before splitting
            X_train_full = X_train_full.sort_index()
            y_train_full = y_train_full.loc[X_train_full.index]

            # Train/val split (80/20 chronological — no shuffle)
            split_idx = int(len(X_train_full) * 0.8)
            X_train, X_val = X_train_full.iloc[:split_idx], X_train_full.iloc[split_idx:]
            y_train, y_val = y_train_full.iloc[:split_idx], y_train_full.iloc[split_idx:]

            # Step 4: Prepare PREDICTION row — use TODAY's data (most recent row in full df)
            # FIXED: Predict on the LATEST actual row, not the last labeled row.
            # The last labeled row is `horizon` days old; today's features are already available.
            X_full = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_latest = X_full.iloc[[-1]]  # Most recent row — TODAY

            # Step 5: Train models (simplified for speed)
            models = ModelManager.train_complete(
                X_train, y_train,
                X_val=X_val, y_val=y_val,
                use_moe=False,  # Disable for speed
                use_tabnet=False,  # Disable for speed
                use_attention=False,
                use_nas=False,
                use_flaml=False
            )

            # Step 6: Get latest prediction (on TODAY's features)
            
            predictions, confidence, proba, info = ModelManager.predict_complete(
                models, X_latest,
                use_moe=False,
                use_tabnet=False,
                use_conformal=False
            )
            
            # Step 7: Convert to signal
            pred = predictions[0]
            conf = confidence[0] if isinstance(confidence, np.ndarray) else confidence
            
            signal = 'BUY' if pred == 1 else 'SELL' if pred == 0 else 'HOLD'
            
            # Adjust confidence by data quality
            quality_modifier = result['data_quality'] / 100.0
            adjusted_confidence = conf * 100 * quality_modifier
            
            result['signal'] = signal
            result['confidence'] = round(adjusted_confidence, 1)
            result['raw_confidence'] = round(conf * 100, 1)
            
            # Calculate expected return (simplified)
            if signal == 'BUY':
                result['expected_return'] = round(threshold * 100, 2)  # +2%
            elif signal == 'SELL':
                result['expected_return'] = round(-threshold * 100, 2)  # -2%
            else:
                result['expected_return'] = 0
            
            # Risk-reward ratio (simplified)
            if result['confidence'] > 0:
                result['risk_reward'] = round(abs(result['expected_return']) / (100 - result['confidence']) * 10, 2)
            
            print(f"  {symbol}: {signal} @ {result['confidence']:.0f}% (Quality: {result['data_quality']:.0f})")
            
        except Exception as e:
            result['error'] = str(e)[:100]
            print(f"  {symbol}: ERROR - {str(e)[:50]}")
        
        return result
    
    def run_analysis(self, stocks: list = None, max_stocks: int = None) -> pd.DataFrame:
        """
        Run analysis on all stocks.
        
        Args:
            stocks: List of stock symbols (or None for all)
            max_stocks: Maximum number of stocks to analyze
        
        Returns:
            DataFrame with all results
        """
        if stocks is None:
            stocks = self.get_stock_universe()
        
        if max_stocks:
            stocks = stocks[:max_stocks]
        
        print(f"\nAnalyzing {len(stocks)} stocks...")
        print("-" * 70)
        
        for i, symbol in enumerate(stocks, 1):
            print(f"[{i}/{len(stocks)}] ", end='')
            result = self.analyze_single_stock(symbol)
            self.results.append(result)
            
            if result['error']:
                self.errors.append({'symbol': symbol, 'error': result['error']})
        
        # Convert to DataFrame
        df_results = pd.DataFrame(self.results)
        
        # Sort by confidence
        df_results = df_results.sort_values('confidence', ascending=False)
        
        return df_results
    
    def generate_report(self, df_results: pd.DataFrame, format: str = 'all') -> dict:
        """
        Generate output reports.
        
        Args:
            df_results: Results DataFrame
            format: 'json', 'csv', 'markdown', 'all'
        
        Returns:
            Dict with file paths
        """
        today = date.today().strftime('%Y-%m-%d')
        files = {}
        
        # Filter valid results
        valid_results = df_results[df_results['error'].isna()].copy()
        
        # Top signals
        buy_signals = valid_results[valid_results['signal'] == 'BUY'].head(10)
        sell_signals = valid_results[valid_results['signal'] == 'SELL'].head(10)
        
        # JSON Report
        if format in ['json', 'all']:
            json_path = self.output_dir / f"signals_{today}.json"
            report = {
                'generated_at': datetime.now().isoformat(),
                'total_stocks': len(df_results),
                'valid_predictions': len(valid_results),
                'errors': len(self.errors),
                'summary': {
                    'buy_signals': len(valid_results[valid_results['signal'] == 'BUY']),
                    'sell_signals': len(valid_results[valid_results['signal'] == 'SELL']),
                    'hold_signals': len(valid_results[valid_results['signal'] == 'HOLD']),
                    'avg_confidence': round(valid_results['confidence'].mean(), 1),
                    'avg_data_quality': round(valid_results['data_quality'].mean(), 1)
                },
                'top_buys': buy_signals.to_dict('records'),
                'top_sells': sell_signals.to_dict('records'),
                'all_signals': valid_results.to_dict('records'),
                'errors': self.errors
            }
            with open(json_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            files['json'] = str(json_path)
        
        # CSV Report
        if format in ['csv', 'all']:
            csv_path = self.output_dir / f"signals_{today}.csv"
            valid_results.to_csv(csv_path, index=False)
            files['csv'] = str(csv_path)
        
        # Markdown Report (for easy reading)
        if format in ['markdown', 'all']:
            md_path = self.output_dir / f"DAILY_SIGNALS_{today}.md"
            
            md_content = f"""# Daily Trading Signals - {today}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')}

## Summary

| Metric | Value |
|--------|-------|
| Total Stocks Analyzed | {len(df_results)} |
| Valid Predictions | {len(valid_results)} |
| Errors | {len(self.errors)} |
| Average Confidence | {valid_results['confidence'].mean():.1f}% |
| Average Data Quality | {valid_results['data_quality'].mean():.1f}% |

## Top BUY Signals

| Stock | Confidence | Data Quality | Expected Return |
|-------|------------|--------------|-----------------|
"""
            for _, row in buy_signals.iterrows():
                md_content += f"| {row['symbol']} | {row['confidence']:.0f}% | {row['data_quality']:.0f}% | {row['expected_return']}% |\n"
            
            md_content += """
## Top SELL Signals

| Stock | Confidence | Data Quality | Expected Return |
|-------|------------|--------------|-----------------|
"""
            for _, row in sell_signals.iterrows():
                md_content += f"| {row['symbol']} | {row['confidence']:.0f}% | {row['data_quality']:.0f}% | {row['expected_return']}% |\n"
            
            md_content += f"""
## Data Quality Warnings

Stocks with data quality below 70% should be traded with caution or skipped entirely.

| Stock | Quality Score | Issues |
|-------|--------------|--------|
"""
            low_quality = valid_results[valid_results['data_quality'] < 70]
            for _, row in low_quality.head(10).iterrows():
                md_content += f"| {row['symbol']} | {row['data_quality']:.0f}% | Check data freshness |\n"
            
            if self.errors:
                md_content += f"""
## Errors ({len(self.errors)} stocks)

| Stock | Error |
|-------|-------|
"""
                for err in self.errors[:10]:
                    md_content += f"| {err['symbol']} | {err['error'][:50]}... |\n"
            
            md_content += """
---
*This report was generated automatically. Always verify signals before trading.*
*Data quality score affects confidence - low quality = lower confidence.*
"""
            
            with open(md_path, 'w', encoding='utf-8') as f:
                f.write(md_content)
            files['markdown'] = str(md_path)
        
        return files
    
    def run(self, stocks: list = None, max_stocks: int = None, output_format: str = 'all'):
        """
        Main entry point.
        
        Args:
            stocks: List of stock symbols
            max_stocks: Max stocks to analyze
            output_format: Report format
        """
        print("\n" + "=" * 70)
        print("STARTING ANALYSIS")
        print("=" * 70)
        
        # Run analysis
        df_results = self.run_analysis(stocks, max_stocks)
        
        # Generate reports
        print("\n" + "-" * 70)
        print("Generating reports...")
        files = self.generate_report(df_results, output_format)
        
        # Print summary
        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        
        valid = df_results[df_results['error'].isna()]
        
        print(f"\nTotal: {len(df_results)} stocks")
        print(f"Valid: {len(valid)} predictions")
        print(f"Errors: {len(self.errors)} stocks")
        
        if len(valid) > 0:
            print(f"\nSignal Distribution:")
            print(f"  BUY:  {len(valid[valid['signal'] == 'BUY'])}")
            print(f"  SELL: {len(valid[valid['signal'] == 'SELL'])}")
            print(f"  HOLD: {len(valid[valid['signal'] == 'HOLD'])}")
            
            print(f"\nAverage Confidence: {valid['confidence'].mean():.1f}%")
            print(f"Average Data Quality: {valid['data_quality'].mean():.1f}%")
        
        print(f"\nReports saved to:")
        for fmt, path in files.items():
            print(f"  {fmt}: {path}")
        
        return df_results, files


def main():
    parser = argparse.ArgumentParser(description='Run daily stock analysis')
    parser.add_argument('--stocks', nargs='+', help='Specific stocks to analyze')
    parser.add_argument('--top', type=int, help='Analyze top N stocks only')
    parser.add_argument('--output', choices=['json', 'csv', 'markdown', 'all'], 
                       default='all', help='Output format')
    parser.add_argument('--output-dir', default='daily_reports', help='Output directory')
    
    args = parser.parse_args()
    
    runner = DailyAnalysisRunner(output_dir=args.output_dir)
    
    results, files = runner.run(
        stocks=args.stocks,
        max_stocks=args.top,
        output_format=args.output
    )
    
    return results


if __name__ == "__main__":
    main()

