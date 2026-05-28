"""
Quick Data Updater
==================
Refreshes OHLCV parquet files for the 70-stock trading universe.
Runs in ~2-3 minutes using yfinance.
Called by RUN_DAILY_SIGNALS.bat before generating signals.

Usage:
    python quick_update_data.py
"""

import time
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

warnings.filterwarnings('ignore')

DATA_DIR = Path("data/stocks")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 70-stock trading universe (Nifty50 + NiftyNext50 minus broken tickers)
UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "AXISBANK.NS", "TITAN.NS", "ULTRACEMCO.NS", "SUNPHARMA.NS", "NESTLEIND.NS",
    "BAJAJFINSV.NS", "WIPRO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS",
    "COALINDIA.NS", "JSWSTEEL.NS", "TATASTEEL.NS", "INDUSINDBK.NS", "TECHM.NS",
    "HINDALCO.NS", "GRASIM.NS", "TATACONSUM.NS", "DRREDDY.NS", "BRITANNIA.NS",
    "CIPLA.NS", "APOLLOHOSP.NS", "BPCL.NS", "LTIM.NS", "SBILIFE.NS",
    "HDFCLIFE.NS", "TRENT.NS", "GAIL.NS", "PIDILITIND.NS", "DABUR.NS",
    "SIEMENS.NS", "DLF.NS", "ICICIPRULI.NS", "BANKBARODA.NS", "CHOLAFIN.NS",
    "ZOMATO.NS", "ABB.NS", "BOSCHLTD.NS", "COLPAL.NS", "MARICO.NS",
    "BEL.NS", "HDFCAMC.NS", "PFC.NS", "RECLTD.NS", "TATAPOWER.NS",
    "HAL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "TATAMTRDVR.NS", "DMART.NS",
    "PGHH.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS",
]

# Also update index data
INDICES = ["^NSEI", "^NSEBANK", "INDIAVIX.NS"]
INDEX_MAP = {"^NSEI": "NIFTY50", "^NSEBANK": "NIFTYBANK", "INDIAVIX.NS": "INDIAVIX"}


def _get_last_date(symbol: str) -> date:
    """Get the last date in the parquet file."""
    nse = symbol.replace(".NS", "").replace("^", "")
    fname = INDEX_MAP.get(symbol, nse)
    fpath = DATA_DIR / f"{fname}.parquet"
    if fpath.exists():
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            return df.index.max().date()
        except Exception:
            pass
    return date.today() - timedelta(days=365)  # Full year if no file


def _update_symbol(symbol: str) -> bool:
    """Download and append latest data for one symbol."""
    nse = symbol.replace(".NS", "").replace("^", "")
    fname = INDEX_MAP.get(symbol, nse)
    fpath = DATA_DIR / f"{fname}.parquet"

    last_date = _get_last_date(symbol)
    today     = date.today()

    if last_date >= today - timedelta(days=1):
        return True  # Already fresh

    start = last_date - timedelta(days=5)  # Small overlap for safety

    try:
        new_df = yf.download(
            symbol,
            start=start.strftime('%Y-%m-%d'),
            end=(today + timedelta(days=1)).strftime('%Y-%m-%d'),
            progress=False,
            auto_adjust=False,
        )

        if new_df.empty:
            return False

        # Flatten MultiIndex columns if present
        if isinstance(new_df.columns, pd.MultiIndex):
            new_df.columns = new_df.columns.get_level_values(0)

        new_df.index = pd.to_datetime(new_df.index)

        # Merge with existing
        if fpath.exists():
            try:
                existing = pd.read_parquet(fpath)
                existing.index = pd.to_datetime(existing.index)
                combined = pd.concat([existing, new_df])
                combined = combined[~combined.index.duplicated(keep='last')]
                combined = combined.sort_index()
            except Exception:
                combined = new_df
        else:
            combined = new_df

        # Keep only OHLCV + Volume columns
        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                if c in combined.columns]
        combined = combined[keep]

        # Drop rows where Close is NaN (incomplete/in-progress trading day)
        combined = combined.dropna(subset=['Close'])
        combined.to_parquet(fpath)
        return True

    except Exception:
        return False


def run_update(verbose: bool = True):
    """Update all universe stocks."""
    today     = date.today()
    all_syms  = UNIVERSE + INDICES
    updated   = 0
    skipped   = 0
    failed    = 0

    if verbose:
        print(f"Updating market data for {len(UNIVERSE)} stocks...")
        print(f"Target date: {today}")

    # Batch download is faster than one-by-one for many symbols
    # But we process in small batches to handle failures gracefully
    batch_size = 10
    batches = [all_syms[i:i+batch_size] for i in range(0, len(all_syms), batch_size)]

    for batch_num, batch in enumerate(batches, 1):
        # Check which symbols in this batch actually need updating
        need_update = []
        for sym in batch:
            last = _get_last_date(sym)
            if last < today - timedelta(days=1):
                need_update.append(sym)
            else:
                skipped += 1

        if not need_update:
            continue

        if verbose:
            pct = int((batch_num / len(batches)) * 100)
            print(f"  Batch {batch_num}/{len(batches)} ({pct}%) — "
                  f"updating {len(need_update)} symbols...", end='\r')

        # Try batch download first (fast)
        try:
            start_date = (today - timedelta(days=10)).strftime('%Y-%m-%d')
            end_date   = (today + timedelta(days=1)).strftime('%Y-%m-%d')

            batch_df = yf.download(
                need_update,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=False,
                group_by='ticker',
            )

            if not batch_df.empty:
                for sym in need_update:
                    nse   = sym.replace(".NS", "").replace("^", "")
                    fname = INDEX_MAP.get(sym, nse)
                    fpath = DATA_DIR / f"{fname}.parquet"
                    try:
                        if len(need_update) == 1:
                            sym_df = batch_df.copy()
                        else:
                            sym_df = batch_df[sym].copy() if sym in batch_df.columns.get_level_values(0) else pd.DataFrame()

                        if sym_df.empty:
                            failed += 1
                            continue

                        sym_df.index = pd.to_datetime(sym_df.index)
                        sym_df = sym_df.dropna(how='all')

                        if fpath.exists():
                            existing = pd.read_parquet(fpath)
                            existing.index = pd.to_datetime(existing.index)
                            combined = pd.concat([existing, sym_df])
                            combined = combined[~combined.index.duplicated(keep='last')]
                            combined = combined.sort_index()
                        else:
                            combined = sym_df

                        keep = [c for c in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
                                if c in combined.columns]
                        # Drop rows where Close is NaN (incomplete trading day)
                        combined = combined.dropna(subset=['Close'])
                        combined[keep].to_parquet(fpath)
                        updated += 1

                    except Exception:
                        # Fall back to individual download
                        ok = _update_symbol(sym)
                        if ok:
                            updated += 1
                        else:
                            failed += 1
            else:
                failed += len(need_update)

        except Exception:
            # Fall back to individual downloads
            for sym in need_update:
                ok = _update_symbol(sym)
                if ok:
                    updated += 1
                else:
                    failed += 1

        time.sleep(0.3)  # Rate limit

    if verbose:
        print(f"  Data update complete:  "
              f"{updated} updated, {skipped} already current, {failed} failed")

        # Verify a sample stock
        sample = "NIFTY50"
        sp = DATA_DIR / f"{sample}.parquet"
        if sp.exists():
            df = pd.read_parquet(sp)
            df.index = pd.to_datetime(df.index)
            last = df.index.max().date()
            close = df['Close'].iloc[-1]
            print(f"  Nifty50 last date: {last}, close: {close:,.0f}")


if __name__ == '__main__':
    run_update()
    # Cleanup temp test file if exists
    import os
    for f in ['_check_prices.py']:
        try:
            os.remove(f)
        except Exception:
            pass
