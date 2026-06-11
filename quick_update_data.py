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

# Dynamic universe: Nifty 500 with liquidity floor (see universe.py).
# Falls back to the legacy 70-stock list if the NSE list is unreachable.
try:
    from universe import get_universe
    UNIVERSE = get_universe(verbose=True)
except Exception as _e:
    print(f"  [WARN] universe module failed ({_e}); using minimal fallback")
    UNIVERSE = [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS",
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
            valid = df.index.dropna()
            if len(valid) > 0:
                last = valid.max()
                if pd.notna(last):
                    return last.date()
        except Exception:
            pass
    # ~800 calendar days (~550 trading days) on cold start so momentum
    # alphas (need 252 trading days) and others have ample history.
    return date.today() - timedelta(days=800)


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
        earliest = today  # earliest last-date in batch -> drives deep cold-start fetch
        for sym in batch:
            last = _get_last_date(sym)
            if last < today - timedelta(days=1):
                need_update.append(sym)
                if last < earliest:
                    earliest = last
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
            # Cold stocks (missing parquet) report last-date = today-800 via
            # _get_last_date, so this fetches deep history for them while warm
            # stocks only pull the recent incremental window.
            start_date = (earliest - timedelta(days=5)).strftime('%Y-%m-%d')
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


def update_from_bhav(verbose: bool = True) -> int:
    """
    Update parquet files using NSE bhav copy (same-day close prices).

    NSE publishes the official bhav copy by ~6 PM IST on the trading day.
    This is far more reliable than yfinance for same-day prices.

    Returns number of stocks updated.
    """
    try:
        from nse_fetcher import get_fetcher
        fetcher = get_fetcher()
    except ImportError:
        if verbose:
            print("  nse_fetcher not available, skipping bhav update")
        return 0

    if verbose:
        print("  Fetching NSE bhav copy for latest close prices...")

    bhav = fetcher.get_delivery_bhav()
    if bhav is None or bhav.empty:
        if verbose:
            print("  Bhav copy not available yet (try after 6 PM IST)")
        return 0

    # The bhav copy date = what date does this data represent?
    bhav_cache_files = sorted(Path("nse_cache").glob("bhav_*.parquet"))
    if bhav_cache_files:
        # Filename is bhav_YYYYMMDD.parquet
        bhav_date_str = bhav_cache_files[-1].stem.replace("bhav_", "")
        try:
            from datetime import datetime
            bhav_date = datetime.strptime(bhav_date_str, "%Y%m%d").date()
        except Exception:
            bhav_date = date.today() - timedelta(days=1)
    else:
        bhav_date = date.today() - timedelta(days=1)

    if verbose:
        print(f"  Bhav copy date: {bhav_date}  ({len(bhav)} stocks)")

    updated = 0
    bhav_date_ts = pd.Timestamp(bhav_date)

    for sym in UNIVERSE:
        nse_sym = sym.replace(".NS", "").upper()
        row = bhav[bhav['SYMBOL'] == nse_sym]
        if row.empty:
            continue

        close_px = float(row['CLOSE'].iloc[0]) if 'CLOSE' in row.columns else None
        if close_px is None or close_px <= 0:
            continue

        fpath = DATA_DIR / f"{nse_sym}.parquet"
        if not fpath.exists():
            continue

        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)

            # Skip if this date is already in the file WITH a valid close
            if bhav_date_ts in df.index:
                try:
                    existing_close = df.loc[bhav_date_ts, 'Close']
                    # Handle both scalar and Series (MultiIndex edge case)
                    if hasattr(existing_close, '__len__'):
                        existing_close = existing_close.iloc[0]
                    if pd.notna(existing_close) and float(existing_close) > 0:
                        continue  # Valid close already present, nothing to do
                    # Date exists but close is NaN/zero — remove it so bhav can fill it
                    df = df[df.index != bhav_date_ts]
                except Exception:
                    df = df[df.index != bhav_date_ts]

            # Build a new row from bhav data
            new_row = pd.DataFrame({
                'Open':      [float(row['OPEN'].iloc[0])   if 'OPEN'   in row.columns else close_px],
                'High':      [float(row['HIGH'].iloc[0])   if 'HIGH'   in row.columns else close_px],
                'Low':       [float(row['LOW'].iloc[0])    if 'LOW'    in row.columns else close_px],
                'Close':     [close_px],
                'Adj Close': [close_px],
                'Volume':    [float(row['TOTTRDQTY'].iloc[0]) if 'TOTTRDQTY' in row.columns else 0],
            }, index=[bhav_date_ts])

            combined = pd.concat([df, new_row])
            combined = combined[~combined.index.duplicated(keep='last')].sort_index()
            combined.to_parquet(fpath)
            updated += 1

        except Exception:
            continue

    if verbose:
        print(f"  Bhav update: {updated} stocks updated with {bhav_date} close prices")

    return updated


def run_update(verbose: bool = True):
    """
    Update all universe stocks.
    Strategy:
      1. Try NSE bhav copy first (same-day, available after 6 PM IST)
      2. Fill gaps with yfinance (historical data, available next morning)
    """
    today = date.today()

    if verbose:
        print(f"Updating market data for {len(UNIVERSE)} stocks...")
        print(f"Target date: {today}")

    # Step 1: Try bhav copy for same-day prices (works after 6 PM)
    bhav_updated = update_from_bhav(verbose=verbose)

    # Step 2: yfinance for any remaining gaps / historical data
    all_syms  = UNIVERSE + INDICES
    updated   = 0
    skipped   = 0
    failed    = 0

    batch_size = 10
    batches = [all_syms[i:i+batch_size] for i in range(0, len(all_syms), batch_size)]

    for batch_num, batch in enumerate(batches, 1):
        need_update = []
        earliest = today  # earliest last-date in batch -> drives deep cold-start fetch
        for sym in batch:
            last = _get_last_date(sym)
            if last < today - timedelta(days=1):
                need_update.append(sym)
                if last < earliest:
                    earliest = last
            else:
                skipped += 1

        if not need_update:
            continue

        if verbose:
            pct = int((batch_num / len(batches)) * 100)
            print(f"  yfinance batch {batch_num}/{len(batches)} ({pct}%) - "
                  f"updating {len(need_update)} symbols...", end='\r')

        try:
            # Cold stocks (missing parquet) report last-date = today-800 via
            # _get_last_date, so this fetches deep history for them while warm
            # stocks only pull the recent incremental window.
            start_date = (earliest - timedelta(days=5)).strftime('%Y-%m-%d')
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
                            lvl0 = batch_df.columns.get_level_values(0)
                            sym_df = batch_df[sym].copy() if sym in lvl0 else pd.DataFrame()

                        if sym_df.empty:
                            failed += 1
                            continue

                        sym_df.index = pd.to_datetime(sym_df.index)
                        sym_df = sym_df.dropna(how='all')
                        sym_df = sym_df.dropna(subset=['Close'])  # Drop incomplete days

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
                        combined[keep].to_parquet(fpath)
                        updated += 1

                    except Exception:
                        ok = _update_symbol(sym)
                        if ok:
                            updated += 1
                        else:
                            failed += 1
            else:
                failed += len(need_update)

        except Exception:
            for sym in need_update:
                ok = _update_symbol(sym)
                if ok:
                    updated += 1
                else:
                    failed += 1

        time.sleep(0.3)

    if verbose:
        total_yf = updated + skipped
        print(f"  yfinance: {updated} updated, {skipped} already current, {failed} failed")

        # Verify
        sp = DATA_DIR / "NIFTY50.parquet"
        if sp.exists():
            df = pd.read_parquet(sp)
            df.index = pd.to_datetime(df.index)
            valid = df['Close'].dropna()
            if not valid.empty:
                last = valid.index[-1].date()
                close = valid.iloc[-1]
                print(f"  Nifty50 last valid date: {last}, close: {close:,.0f}")

        # Final check: what's the freshest date across all stocks?
        dates = []
        for sym in ['RELIANCE', 'ONGC', 'HDFCBANK', 'INFY', 'BHARTIARTL']:
            p = DATA_DIR / f"{sym}.parquet"
            if p.exists():
                try:
                    d = pd.read_parquet(p)
                    d.index = pd.to_datetime(d.index)
                    valid_dates = d['Close'].dropna().index
                    if len(valid_dates) > 0:
                        dates.append(valid_dates.max().date())
                except Exception:
                    pass
        if dates:
            freshest = max(dates)
            print(f"  Freshest data across key stocks: {freshest}")
            if freshest < today - timedelta(days=1):
                print(f"  NOTE: Data is {(today - freshest).days} day(s) old.")
                print(f"  If before 6 PM IST: this is normal (bhav not published yet).")
                print(f"  If after 6 PM IST: bhav copy may not be accessible today.")


if __name__ == '__main__':
    run_update()
    import os
    for f in ['_check_prices.py', '_debug_date.py']:
        try:
            os.remove(f)
        except Exception:
            pass
