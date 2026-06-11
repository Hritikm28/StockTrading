"""
Stock Universe Builder
======================
Expands the trading universe from Nifty 100 large-caps (the most efficiently
priced, most institutionally crowded stocks in India) to the Nifty 500 —
where mid/small-caps live and where a small trader's edge actually exists:
funds managing thousands of crores cannot build positions in a Rs 2,000cr
market-cap stock without moving it. We can.

Safety: a LIQUIDITY FLOOR (median daily traded value) guarantees every stock
in the universe can be entered AND exited at retail size without slippage
blowing past the cost model.

Flow:
  1. Fetch the official Nifty 500 constituent list from NSE archives
     (free CSV, cached 7 days, static fallback = legacy 70-stock universe).
  2. Rank candidates by median daily traded value over the last 60 sessions
     (from local parquet data; unranked new symbols are appended so their
     data gets bootstrapped and they qualify on later runs).
  3. Apply the floor, cap at UNIVERSE_SIZE.

Usage:
    from universe import get_universe, get_name_to_symbol_map
    symbols = get_universe()           # ['RELIANCE.NS', ...]
"""

import io
import os
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

UNIVERSE_DIR = Path("data/universe")
UNIVERSE_DIR.mkdir(parents=True, exist_ok=True)
NIFTY500_CACHE = UNIVERSE_DIR / "nifty500.csv"
CACHE_MAX_AGE_DAYS = 7

DATA_DIR = Path("data/stocks")

# Tunables (env-overridable so the cloud workflow can adjust without edits)
UNIVERSE_SIZE = int(os.environ.get('STOCK_UNIVERSE_SIZE', '150'))
MIN_MEDIAN_TRADED_VALUE_INR = float(
    os.environ.get('MIN_TRADED_VALUE_CR', '5')) * 1e7   # default ₹5 crore/day
LIQUIDITY_LOOKBACK_SESSIONS = 60

_NIFTY500_URLS = [
    "https://archives.nseindia.com/content/indices/ind_nifty500list.csv",
    "https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv",
]

_HEADERS = {
    'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                   'AppleWebKit/537.36 (KHTML, like Gecko) '
                   'Chrome/124.0.0.0 Safari/537.36'),
    'Referer': 'https://www.nseindia.com/',
}

# Legacy 70-stock universe — fallback if the NSE list is unreachable and no
# cache exists. Keeps the daily run alive no matter what.
FALLBACK_UNIVERSE = [
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
    "HAL.NS", "ADANIENT.NS", "ADANIPORTS.NS", "DMART.NS",
    "PGHH.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "HEROMOTOCO.NS", "M&M.NS",
]


def fetch_nifty500_list(force: bool = False) -> Optional[pd.DataFrame]:
    """
    Official Nifty 500 constituents (Company Name, Industry, Symbol, ISIN).
    Cached locally for CACHE_MAX_AGE_DAYS.
    """
    if not force and NIFTY500_CACHE.exists():
        try:
            age_days = (time.time() - NIFTY500_CACHE.stat().st_mtime) / 86400
            if age_days < CACHE_MAX_AGE_DAYS:
                return pd.read_csv(NIFTY500_CACHE)
        except Exception:
            pass

    for url in _NIFTY500_URLS:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=25)
            if resp.status_code != 200 or 'Symbol' not in resp.text[:500]:
                continue
            df = pd.read_csv(io.StringIO(resp.text))
            df.columns = [c.strip() for c in df.columns]
            if 'Symbol' not in df.columns:
                continue
            df['Symbol'] = df['Symbol'].astype(str).str.strip().str.upper()
            df = df[df['Symbol'].str.len() > 0]
            try:
                df.to_csv(NIFTY500_CACHE, index=False)
            except Exception:
                pass
            return df
        except Exception:
            continue

    # Stale cache beats nothing
    if NIFTY500_CACHE.exists():
        try:
            return pd.read_csv(NIFTY500_CACHE)
        except Exception:
            pass
    return None


def _median_traded_value(symbol: str) -> Optional[float]:
    """Median daily traded value (₹) over recent sessions from local parquet."""
    nse = symbol.replace('.NS', '').upper()
    fpath = DATA_DIR / f"{nse}.parquet"
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath, columns=['Close', 'Volume'])
        df = df.dropna().tail(LIQUIDITY_LOOKBACK_SESSIONS)
        if len(df) < 20:
            return None
        return float((df['Close'] * df['Volume']).median())
    except Exception:
        return None


def get_universe(max_size: int = None,
                 min_traded_value: float = None,
                 verbose: bool = False) -> List[str]:
    """
    Returns the trading universe as ['SYMBOL.NS', ...].

    Liquid known names ranked by traded value come first; symbols without
    local data yet are appended (they get bootstrapped by quick_update_data
    and earn a liquidity rank on subsequent runs).
    """
    max_size = max_size or UNIVERSE_SIZE
    min_tv = min_traded_value if min_traded_value is not None \
        else MIN_MEDIAN_TRADED_VALUE_INR

    n500 = fetch_nifty500_list()
    if n500 is None or n500.empty:
        if verbose:
            print("   [universe] NSE list unavailable — using fallback "
                  f"{len(FALLBACK_UNIVERSE)}-stock universe")
        return FALLBACK_UNIVERSE[:max_size]

    candidates = [f"{s}.NS" for s in n500['Symbol'].tolist()]

    ranked, unknown = [], []
    for sym in candidates:
        tv = _median_traded_value(sym)
        if tv is None:
            unknown.append(sym)
        elif tv >= min_tv:
            ranked.append((sym, tv))
        # below the floor → excluded outright

    ranked.sort(key=lambda x: -x[1])
    universe = [s for s, _ in ranked]

    # Append data-less candidates so they get bootstrapped, capped so a cold
    # start doesn't try to download 500 deep histories in one run.
    room = max(max_size - len(universe), 0)
    universe += unknown[:min(room + 25, len(unknown))]

    universe = universe[:max_size]

    if verbose:
        print(f"   [universe] {len(universe)} stocks "
              f"({len(ranked)} liquidity-ranked >= ₹{min_tv/1e7:.0f}cr/day, "
              f"{len(universe) - min(len(ranked), max_size)} bootstrapping)")
    if not universe:
        return FALLBACK_UNIVERSE[:max_size]
    return universe


def get_name_to_symbol_map() -> Dict[str, str]:
    """
    {normalized company name: SYMBOL} from the Nifty 500 list.
    Used by alphas whose NSE feeds key on company name (e.g. pledge data).
    """
    n500 = fetch_nifty500_list()
    if n500 is None or n500.empty:
        return {}
    name_col = next((c for c in n500.columns
                     if c.lower().startswith('company')), None)
    if name_col is None:
        return {}
    out = {}
    for _, r in n500.iterrows():
        name = normalize_company_name(str(r[name_col]))
        if name:
            out[name] = str(r['Symbol']).upper()
    return out


def normalize_company_name(name: str) -> str:
    """Lowercase, strip 'Limited/Ltd' suffixes and punctuation for matching."""
    n = name.lower().strip()
    for suffix in [' limited', ' ltd.', ' ltd', ' (india)', ' india limited']:
        if n.endswith(suffix):
            n = n[: -len(suffix)]
    return ''.join(ch for ch in n if ch.isalnum() or ch == ' ').strip()


if __name__ == '__main__':
    u = get_universe(verbose=True)
    print(f"Universe ({len(u)}): first 20 -> {u[:20]}")
