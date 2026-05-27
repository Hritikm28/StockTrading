"""
NSE Data Fetcher
================
Session-aware fetcher for all NSE free APIs.

Handles:
  - Cookie-based session initialization (visit homepage first)
  - Option chain (PCR, Max Pain, OI by strike)
  - Corporate announcements
  - Delivery % from bhav copy (EQ series)
  - F&O ban list

All data is cached locally to avoid rate-limiting.

Usage:
    from nse_fetcher import get_fetcher
    f = get_fetcher()
    oc  = f.get_option_chain("RELIANCE")
    ann = f.get_announcements("RELIANCE")
    bhav = f.get_delivery_bhav()   # latest trading day
"""

import io
import json
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Dict, List

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CACHE_DIR = Path("nse_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

NSE_BASE = "https://www.nseindia.com"

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    ),
    'Accept':          '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection':      'keep-alive',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _compute_max_pain(strikes: dict, strike_prices: list) -> float:
    """
    Max Pain = strike where total option loss to buyers is maximized.
    Market often pins near this price on expiry.
    """
    if not strike_prices:
        return 0.0

    min_pain   = float('inf')
    best_strike = strike_prices[0]

    for test in strike_prices:
        ce_pain = sum(max(0, s - test) * v['ce_oi'] for s, v in strikes.items())
        pe_pain = sum(max(0, test - s) * v['pe_oi'] for s, v in strikes.items())
        total   = ce_pain + pe_pain
        if total < min_pain:
            min_pain    = total
            best_strike = test

    return float(best_strike)


# ---------------------------------------------------------------------------
# NSEFetcher class
# ---------------------------------------------------------------------------
class NSEFetcher:
    """
    Session-aware NSE data fetcher.
    Single instance is reused (singleton via get_fetcher()).
    """

    def __init__(self, cache_hours: int = 4):
        self.session      = requests.Session()
        self.session.headers.update(_HEADERS)
        self.cache_hours  = cache_hours
        self._initialized = False

    # ------------------------------------------------------------------ #
    #  Session management                                                  #
    # ------------------------------------------------------------------ #
    def _init_session(self):
        """
        Two-step NSE session initialization:
          1. Visit homepage to get base cookies
          2. Visit option-chain page to get market-data cookies
        This is required for NSE API calls to return 200.
        """
        if self._initialized:
            return
        try:
            # Step 1: Homepage
            self.session.get(
                NSE_BASE,
                headers={**_HEADERS, 'Referer': ''},
                timeout=15,
            )
            time.sleep(1.0)
            # Step 2: Option chain page (sets additional cookies)
            self.session.get(
                NSE_BASE + '/option-chain',
                headers={**_HEADERS, 'Referer': NSE_BASE + '/'},
                timeout=15,
            )
            time.sleep(0.8)
        except Exception:
            pass  # continue anyway
        self._initialized = True

    def _get(self, url: str, retries: int = 3,
             extra_headers: dict = None) -> Optional[requests.Response]:
        """Authenticated GET with retry + session re-init on 401/403."""
        self._init_session()
        headers = {
            **_HEADERS,
            'Referer': NSE_BASE + '/',
            'X-Requested-With': 'XMLHttpRequest',
        }
        if extra_headers:
            headers.update(extra_headers)
        for attempt in range(retries):
            try:
                resp = self.session.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    return resp
                if resp.status_code in (401, 403):
                    # Re-initialize session and retry
                    self._initialized = False
                    self._init_session()
                    time.sleep(2.0)
            except Exception:
                pass
            time.sleep(1.5 ** attempt)
        return None

    # ------------------------------------------------------------------ #
    #  Disk cache                                                          #
    # ------------------------------------------------------------------ #
    def _cache_path(self, name: str) -> Path:
        return CACHE_DIR / f"{name}.json"

    def _load_cache(self, name: str, max_age_hours: int = None) -> Optional[dict]:
        age_limit = max_age_hours or self.cache_hours
        path = self._cache_path(name)
        if path.exists():
            try:
                age_h = (time.time() - path.stat().st_mtime) / 3600
                if age_h < age_limit:
                    with open(path, encoding='utf-8') as f:
                        return json.load(f)
            except Exception:
                pass
        return None

    def _save_cache(self, name: str, data):
        try:
            with open(self._cache_path(name), 'w', encoding='utf-8') as f:
                json.dump(data, f, default=str)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    #  Option Chain  (yfinance primary, NSE fallback)                     #
    # ------------------------------------------------------------------ #
    def get_option_chain(self, symbol: str) -> Optional[Dict]:
        """
        Fetch option chain via yfinance (reliable) with NSE fallback.

        Returns:
          pcr         : float  (put-call ratio by OI)
          max_pain    : float  (strike with max option buyer loss)
          underlying  : float  (current spot price)
          total_ce_oi : int
          total_pe_oi : int
          strikes     : dict   {strike: {ce_oi, pe_oi, ...}}
        """
        sym       = symbol.replace('.NS', '').upper()
        cache_key = f"oc_{sym}"
        cached    = self._load_cache(cache_key, max_age_hours=2)
        if cached:
            return cached

        # Primary: yfinance (works reliably for NSE stocks)
        result = self._get_option_chain_yf(sym)

        # Fallback: NSE direct API
        if not result:
            result = self._get_option_chain_nse(sym)

        if result:
            self._save_cache(cache_key, result)

        return result

    def _get_option_chain_yf(self, nse_sym: str) -> Optional[Dict]:
        """Fetch option chain via yfinance."""
        try:
            import yfinance as yf
            ticker = yf.Ticker(f"{nse_sym}.NS")
            exps   = ticker.options
            if not exps:
                return None

            # Use nearest expiry (first in list)
            exp = exps[0]
            chain = ticker.option_chain(exp)
            calls = chain.calls
            puts  = chain.puts

            if calls.empty and puts.empty:
                return None

            # Build strikes dict
            all_strikes = sorted(
                set(calls['strike'].tolist()) | set(puts['strike'].tolist())
            )
            strikes: Dict[float, dict] = {}
            for strike in all_strikes:
                c_row = calls[calls['strike'] == strike]
                p_row = puts[puts['strike']  == strike]
                strikes[float(strike)] = {
                    'ce_oi':  int(c_row['openInterest'].iloc[0])  if not c_row.empty else 0,
                    'pe_oi':  int(p_row['openInterest'].iloc[0])  if not p_row.empty else 0,
                    'ce_ltp': float(c_row['lastPrice'].iloc[0])   if not c_row.empty else 0.0,
                    'pe_ltp': float(p_row['lastPrice'].iloc[0])   if not p_row.empty else 0.0,
                    'ce_iv':  float(c_row['impliedVolatility'].iloc[0]) if not c_row.empty else 0.0,
                    'pe_iv':  float(p_row['impliedVolatility'].iloc[0]) if not p_row.empty else 0.0,
                }

            total_ce = sum(v['ce_oi'] for v in strikes.values())
            total_pe = sum(v['pe_oi'] for v in strikes.values())

            if total_ce == 0 and total_pe == 0:
                return None

            pcr = total_pe / total_ce if total_ce > 0 else 1.0

            sp       = sorted(strikes.keys())
            max_pain = _compute_max_pain(strikes, sp)

            # Current price from ticker info
            info       = ticker.fast_info
            underlying = float(getattr(info, 'last_price', 0) or 0)

            return {
                'symbol':      nse_sym,
                'pcr':         round(pcr, 3),
                'max_pain':    max_pain,
                'underlying':  underlying,
                'total_ce_oi': total_ce,
                'total_pe_oi': total_pe,
                'expiry':      exp,
                'source':      'yfinance',
            }
        except Exception as exc:
            return None

    def _get_option_chain_nse(self, nse_sym: str) -> Optional[Dict]:
        """Fetch option chain from NSE API (needs valid session cookies)."""
        try:
            url  = f"{NSE_BASE}/api/option-chain-equities?symbol={nse_sym}"
            resp = self._get(url)
            if not resp:
                return None

            data = resp.json()
            if not data or len(str(data)) < 10:
                return None

            filtered = data.get('filtered', {})
            records  = filtered.get('data', [])
            if not records:
                return None

            strikes: Dict[float, dict] = {}
            for rec in records:
                strike = float(rec.get('strikePrice', 0) or 0)
                ce     = rec.get('CE', {}) or {}
                pe     = rec.get('PE', {}) or {}
                if strike:
                    strikes[strike] = {
                        'ce_oi':  int(ce.get('openInterest', 0) or 0),
                        'pe_oi':  int(pe.get('openInterest', 0) or 0),
                        'ce_ltp': float(ce.get('lastPrice', 0) or 0),
                        'pe_ltp': float(pe.get('lastPrice', 0) or 0),
                    }

            if not strikes:
                return None

            total_ce   = sum(v['ce_oi'] for v in strikes.values())
            total_pe   = sum(v['pe_oi'] for v in strikes.values())
            pcr        = total_pe / total_ce if total_ce > 0 else 1.0
            sp         = sorted(strikes.keys())
            max_pain   = _compute_max_pain(strikes, sp)
            underlying = float(
                data.get('records', {}).get('underlyingValue', 0) or 0
            )

            return {
                'symbol':      nse_sym,
                'pcr':         round(pcr, 3),
                'max_pain':    max_pain,
                'underlying':  underlying,
                'total_ce_oi': total_ce,
                'total_pe_oi': total_pe,
                'source':      'nse',
            }
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    #  Corporate Announcements                                             #
    # ------------------------------------------------------------------ #
    def get_announcements(self, symbol: str) -> List[Dict]:
        """
        Returns list of recent corporate announcements.

        Each item:
          symbol, desc, date, an_date, attachment
        """
        sym       = symbol.replace('.NS', '').upper()
        cache_key = f"ann_{sym}"
        cached    = self._load_cache(cache_key, max_age_hours=6)
        if cached is not None:
            return cached

        url  = (f"{NSE_BASE}/api/corporate-announcements"
                f"?index=equities&symbol={sym}")
        resp = self._get(url)
        if not resp:
            return []

        try:
            data = resp.json()
            if not isinstance(data, list):
                return []

            result = []
            for item in data[:60]:
                result.append({
                    'symbol':     item.get('symbol', sym),
                    'desc':       item.get('desc', ''),
                    'date':       item.get('dt', ''),
                    'an_date':    item.get('an_dt', ''),
                    'attachment': item.get('attchmntText', ''),
                })

            self._save_cache(cache_key, result)
            return result

        except Exception as exc:
            print(f"   [NSEFetcher] Announcements error {sym}: {exc}")
            return []

    # ------------------------------------------------------------------ #
    #  Bhav Copy (Delivery %)                                             #
    # ------------------------------------------------------------------ #
    def get_delivery_bhav(self, as_of_date: date = None) -> Optional[pd.DataFrame]:
        """
        Returns NSE CM bhav copy as DataFrame with columns:
          SYMBOL, DELIV_PER, CLOSE, TOTTRDQTY, TOTTRDVAL

        Tries last 5 calendar days to handle weekends / holidays.
        """
        target = as_of_date or date.today()
        for days_back in range(6):
            d   = target - timedelta(days=days_back)
            df  = self._fetch_bhav_date(d)
            if df is not None and not df.empty:
                return df
        return None

    def _fetch_bhav_date(self, d: date) -> Optional[pd.DataFrame]:
        """Fetch bhav copy for one specific date."""
        cache_file = CACHE_DIR / f"bhav_{d.strftime('%Y%m%d')}.parquet"
        if cache_file.exists():
            try:
                return pd.read_parquet(cache_file)
            except Exception:
                pass

        dd      = d.strftime('%d')
        mm      = d.strftime('%m')
        yyyy    = d.strftime('%Y')
        ddmmyyyy = f"{dd}{mm}{yyyy}"
        mmm     = d.strftime('%b').upper()

        # URL patterns from newest to oldest format
        urls = [
            # 2024+ NSE CM ZIP
            (f"https://archives.nseindia.com/content/cm/"
             f"BhavCopy_NSE_CM_0_0_0_{ddmmyyyy}_F_0000.csv.zip"),
            # Mirror
            (f"https://nsearchives.nseindia.com/content/cm/"
             f"BhavCopy_NSE_CM_0_0_0_{ddmmyyyy}_F_0000.csv.zip"),
            # Pre-2024 full bhav (plain CSV)
            (f"https://archives.nseindia.com/products/content/"
             f"sec_bhavdata_full_{ddmmyyyy}.csv"),
            # Alternative old format by month-year
            (f"https://archives.nseindia.com/content/historical/EQUITIES/"
             f"{yyyy}/{mmm}/cm{dd}{mmm}{yyyy}bhav.csv.zip"),
        ]

        headers = {**_HEADERS, 'Referer': 'https://www.nseindia.com/'}

        for url in urls:
            try:
                resp = self.session.get(url, headers=headers,
                                        timeout=30, stream=True)
                if resp.status_code != 200:
                    continue

                content = resp.content
                df = None

                if url.endswith('.zip'):
                    try:
                        with zipfile.ZipFile(io.BytesIO(content)) as z:
                            csv_names = [n for n in z.namelist()
                                         if n.lower().endswith('.csv')]
                            if csv_names:
                                with z.open(csv_names[0]) as csvf:
                                    df = pd.read_csv(csvf)
                    except Exception:
                        continue
                else:
                    try:
                        df = pd.read_csv(
                            io.StringIO(content.decode('utf-8', errors='ignore'))
                        )
                    except Exception:
                        continue

                if df is None or df.empty:
                    continue

                # Normalize column names
                df.columns = [c.strip().upper().replace(' ', '_')
                               for c in df.columns]

                # Keep EQ series only
                if 'SERIES' in df.columns:
                    df = df[df['SERIES'].str.strip() == 'EQ'].copy()

                # Find delivery column
                deliv_col = None
                for candidate in ['DELIV_PER', 'DELIVERYPERCENTAGE',
                                   'DELIVERY_PER', '%DLYTOTRADED',
                                   'PERCTDLYVSTRD']:
                    if candidate in df.columns:
                        deliv_col = candidate
                        break

                if deliv_col is None:
                    continue  # This URL format doesn't have delivery data

                df = df.rename(columns={deliv_col: 'DELIV_PER'})

                if 'SYMBOL' not in df.columns:
                    continue

                df['SYMBOL']    = df['SYMBOL'].str.strip().str.upper()
                df['DELIV_PER'] = pd.to_numeric(df['DELIV_PER'],
                                                errors='coerce').fillna(0)

                keep = ['SYMBOL', 'DELIV_PER']
                for col in ['CLOSE', 'TOTTRDQTY', 'TOTTRDVAL', 'PREVCLOSE']:
                    if col in df.columns:
                        keep.append(col)

                df = df[keep].reset_index(drop=True)
                df = df[df['DELIV_PER'] > 0]  # Drop rows with no delivery data

                if df.empty:
                    continue

                try:
                    df.to_parquet(cache_file)
                except Exception:
                    pass

                return df

            except Exception:
                continue

        return None

    # ------------------------------------------------------------------ #
    #  F&O Ban List                                                        #
    # ------------------------------------------------------------------ #
    def get_fo_ban_list(self) -> List[str]:
        """
        Returns list of symbols currently in F&O ban.
        Caches for 6 hours.
        """
        cache_key = "fo_ban"
        cached    = self._load_cache(cache_key, max_age_hours=6)
        if cached is not None:
            return cached

        url  = f"{NSE_BASE}/api/fo-ban-list"
        resp = self._get(url)
        if not resp:
            return []

        try:
            data   = resp.json()
            syms   = []
            items  = data.get('data', data) if isinstance(data, dict) else data
            for item in (items if isinstance(items, list) else []):
                s = (item.get('symbol', '') or item.get('SYMBOL', '')
                     if isinstance(item, dict) else str(item))
                if s:
                    syms.append(s.strip().upper())

            self._save_cache(cache_key, syms)
            return syms

        except Exception:
            return []


# ---------------------------------------------------------------------------
# Global singleton
# ---------------------------------------------------------------------------
_fetcher: Optional[NSEFetcher] = None


def get_fetcher() -> NSEFetcher:
    """Return the global NSEFetcher instance (creates once)."""
    global _fetcher
    if _fetcher is None:
        _fetcher = NSEFetcher()
    return _fetcher


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from datetime import date as dt
    print("Testing NSEFetcher...")

    f = get_fetcher()

    print("\n1) Option Chain - RELIANCE")
    oc = f.get_option_chain("RELIANCE")
    if oc:
        print(f"   PCR={oc['pcr']:.2f}  Max Pain={oc['max_pain']:,.0f}"
              f"  Spot={oc['underlying']:,.0f}")
    else:
        print("   [FAILED]")

    print("\n2) Announcements - RELIANCE")
    ann = f.get_announcements("RELIANCE")
    if ann:
        print(f"   Got {len(ann)} announcements")
        print(f"   Latest: {ann[0]['desc'][:80]}")
    else:
        print("   [FAILED]")

    print("\n3) Bhav copy (delivery %)")
    bhav = f.get_delivery_bhav()
    if bhav is not None:
        print(f"   Got {len(bhav)} stocks")
        row = bhav[bhav['SYMBOL'] == 'RELIANCE']
        if not row.empty:
            print(f"   RELIANCE delivery%: {row['DELIV_PER'].iloc[0]:.1f}%")
    else:
        print("   [FAILED] No bhav copy found")

    print("\nDone.")
