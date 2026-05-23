import pandas as pd
import numpy as np
from multiprocessing import Pool, cpu_count
from datetime import datetime, timedelta
import time
from typing import List, Tuple
from data_manager import DataManager
from config import Config

class ParallelStockProcessor:
    """Process multiple stocks in parallel - the RIGHT way"""
    
    def __init__(self, n_workers=None, batch_size=None):

        if n_workers is None:
            # Use Config.N_JOBS if available, otherwise limit to 4
            try:
                n_workers = Config.N_JOBS if hasattr(Config, 'N_JOBS') else 4
            except:
                n_workers = 4
        
        # Cap at 4 workers for stability on laptops
        self.n_workers = min(n_workers, 4)
        
        # Batch size for processing
        if batch_size is None:
            try:
                batch_size = Config.BATCH_SIZE if hasattr(Config, 'BATCH_SIZE') else 20
            except:
                batch_size = 20
        
        self.batch_size = batch_size
        print(f"🚀 Parallel processor initialized: {self.n_workers} workers, batch size: {self.batch_size}")
    
    def process_stocks_parallel(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        compute_features: bool = True
    ) -> dict:

        print(f"📊 Processing {len(symbols)} stocks with {self.n_workers} workers...")
        start_time = time.time()
        
        # Prepare arguments for each worker
        args_list = [
            (symbol, start_date, end_date, compute_features)
            for symbol in symbols
        ]
        
        # Process in parallel
        with Pool(self.n_workers) as pool:
            results = pool.map(_process_single_stock_worker, args_list)
        
        # Convert to dict
        results_dict = {}
        success_count = 0
        
        for symbol, df, error in results:
            if df is not None:
                results_dict[symbol] = df
                success_count += 1
            else:
                print(f"❌ {symbol}: {error}")
        
        elapsed = time.time() - start_time
        print(f"✅ Processed {success_count}/{len(symbols)} stocks in {elapsed:.1f}s")
        print(f"⚡ Speed: {elapsed/len(symbols):.1f}s per stock")
        
        return results_dict
    
    def process_stocks_batch(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        batch_size: int = 50,
        compute_features: bool = True
    ) -> dict:

        all_results = {}
        total_batches = (len(symbols) + batch_size - 1) // batch_size
        
        print(f"📦 Processing {len(symbols)} stocks in {total_batches} batches of {batch_size}")
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            print(f"\n🔄 Batch {batch_num}/{total_batches} ({len(batch)} stocks)...")
            
            batch_results = self.process_stocks_parallel(
                batch, start_date, end_date, compute_features
            )
            
            all_results.update(batch_results)
            
            # Progress
            processed = len(all_results)
            pct = processed / len(symbols) * 100
            print(f"📈 Progress: {processed}/{len(symbols)} ({pct:.1f}%)")
        
        return all_results


# CRITICAL: Worker function must be at module level for Windows compatibility
def _process_single_stock_worker(args):

    symbol, start_date, end_date, compute_features = args
    
    try:
        # Each worker creates its own DataManager
        dm = DataManager()
        
        # Process stock
        df = dm.fetch_stock_data_with_features(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            compute_features=compute_features
        )
        
        return (symbol, df, None)
        
    except Exception as e:
        return (symbol, None, str(e))


# Convenience function for quick usage
def process_stocks_fast(
    symbols: List[str],
    start_date: str,
    end_date: str,
    n_workers: int = None
) -> dict:

    processor = ParallelStockProcessor(n_workers=n_workers)
    return processor.process_stocks_parallel(symbols, start_date, end_date)