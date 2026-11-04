"""
India-Specific Alpha Signal Modules
=====================================
Free, India-only edges that institutional funds largely ignore because
the opportunity size is too small for their AUM.

Modules:
  1. FIIDIIAlpha       — FII/DII flow momentum from NSE daily files
  2. BulkDealAlpha     — Follow-through after bulk/block deals
  3. InsiderAlpha      — SEBI insider buying/selling signals
  4. PEADAlpha         — Post-Earnings Announcement Drift (30-60 day)
  5. FOBanAlpha        — F&O ban list mean-reversion
  6. IndexRebalAlpha   — NIFTY/NEXT50 rebalance front-run
  7. MomentumAlpha     — 12-1 month cross-sectional momentum
  8. MeanRevAlpha      — 5-day RSI mean-reversion for index stocks

Each module:
  - Takes a stock symbol + as_of_date
  - Returns a score in [-1.0, +1.0]  (+1 = strong buy, -1 = strong sell)
  - Returns a confidence 0-100
  - Is completely independent (no shared state)
"""

import pandas as pd
import numpy as np
import requests
import io
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# ── Cache directory ────────────────────────────────────────────────────────────
_CACHE_DIR = Path("data/alpha_cache")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


_NSE_SESSION: Optional[requests.Session] = None


def _nse_session() -> requests.Session:
    """Session pre-warmed on the NSE homepage.

    www.nseindia.com/api/* endpoints reject cookie-less requests from
    datacenter IPs (GitHub Actions) with 401/403 or an HTML error page —
    this is why the API-backed alphas (option_chain, corp_event, sast,
    shp_delta...) scored 0.0 on every cloud run while the static
    nsearchives.nseindia.com files kept working. Hitting the homepage
    first issues the required cookies; reuse the session for all calls.
    """
    global _NSE_SESSION
    if _NSE_SESSION is None:
        s = requests.Session()
        s.headers.update({
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/124.0 Safari/537.36'),
            'Accept': 'application/json,text/html,*/*',
            'Accept-Language': 'en-US,en;q=0.9',
        })
        try:
            s.get('https://www.nseindia.com/', timeout=10)
        except Exception:
            pass
        _NSE_SESSION = s
    return _NSE_SESSION


_GET_FAILS: Dict[str, int] = {}
_GET_BREAKER_LIMIT = 3


def _safe_get(url: str, timeout: int = 10, headers: dict = None) -> Optional[requests.Response]:
    """Robust HTTP GET with retries; NSE API calls go through the
    cookie-warmed session, everything else through plain requests.

    A URL path that fails _GET_BREAKER_LIMIT calls in a row is skipped for
    the rest of the run: several alphas call this once per stock, and ~5s
    of retries per call per stock would push the daily run past its time
    limit whenever a source is down."""
    endpoint = url.split('?', 1)[0]
    if _GET_FAILS.get(endpoint, 0) >= _GET_BREAKER_LIMIT:
        return None
    resp = _get_with_retries(url, timeout, headers)
    _GET_FAILS[endpoint] = 0 if resp is not None else _GET_FAILS.get(endpoint, 0) + 1
    return resp


def _get_with_retries(url: str, timeout: int, headers: Optional[dict]) -> Optional[requests.Response]:
    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json,text/html,*/*'
    }
    if headers:
        _headers.update(headers)
    needs_cookies = 'www.nseindia.com/api' in url
    for attempt in range(3):
        try:
            if needs_cookies:
                resp = _nse_session().get(url, headers=headers or {},
                                          timeout=timeout)
            else:
                resp = requests.get(url, headers=_headers, timeout=timeout)
            if resp.status_code == 200:
                body_head = resp.text[:200].lstrip().lower()
                # NSE serves an HTML block page with HTTP 200 — treat as miss
                if needs_cookies and body_head.startswith(('<!doctype', '<html')):
                    global _NSE_SESSION
                    _NSE_SESSION = None      # force a cookie refresh
                else:
                    return resp
        except Exception:
            if needs_cookies:
                _NSE_SESSION = None
        import time; time.sleep(1.5 ** attempt)
    return None


# ==============================================================================
# 1. FII / DII FLOW ALPHA
# ==============================================================================
class FIIDIIAlpha:
    """
    FII/DII institutional flow momentum.

    NSE publishes daily FII+DII buy/sell data. Sustained positive FII
    net buying (3-5 day rolling) predicts next-day positive returns for
    Nifty 50 stocks. Reverse for DII (they buy when FII sells).

    Edge: Persistent flows → 3-5 day momentum signal.
    Data: NSE https://www.nseindia.com/api/fiidiiTradeReact  (free, JSON)
    """

    _cache_file = _CACHE_DIR / "fii_dii_flows.parquet"
    _cache_date: Optional[date] = None
    _cached_df: Optional[pd.DataFrame] = None

    @classmethod
    def fetch_flows(cls, lookback_days: int = 30) -> Optional[pd.DataFrame]:
        """Fetch FII/DII flow data from NSE."""
        today = date.today()

        # Return cached data if fresh
        if cls._cache_date == today and cls._cached_df is not None:
            return cls._cached_df

        url = "https://www.nseindia.com/api/fiidiiTradeReact"
        resp = _safe_get(url, headers={'Referer': 'https://www.nseindia.com/'})

        if resp is None:
            # Try disk cache
            if cls._cache_file.exists():
                try:
                    df = pd.read_parquet(cls._cache_file)
                    cls._cached_df = df
                    return df
                except Exception:
                    pass
            return None

        try:
            data = resp.json()
            if not data:
                return None

            # NSE's live schema is one row PER CATEGORY per date with string
            # values: {"buyValue":"22779.32","category":"DII","date":"05-Jun-2026",...}
            # (verified by cloud_nse_probe). The old per-date fiiBuyValue
            # schema is kept as fallback. Parsing the wrong schema returned
            # all-zero flows and silently killed this alpha for months.
            by_date: Dict = {}
            for entry in data:
                try:
                    if 'category' in entry:
                        d = pd.to_datetime(entry.get('date', ''))
                        cat = str(entry.get('category', '')).upper()
                        row = by_date.setdefault(d, {
                            'fii_buy': 0.0, 'fii_sell': 0.0,
                            'dii_buy': 0.0, 'dii_sell': 0.0})
                        buy  = float(entry.get('buyValue') or 0)
                        sell = float(entry.get('sellValue') or 0)
                        if 'FII' in cat or 'FPI' in cat:
                            row['fii_buy'], row['fii_sell'] = buy, sell
                        elif 'DII' in cat:
                            row['dii_buy'], row['dii_sell'] = buy, sell
                    else:
                        d = pd.to_datetime(
                            entry.get('date', entry.get('tradeDate', '')))
                        by_date[d] = {
                            'fii_buy':  float(entry.get('fiiBuyValue', 0) or 0),
                            'fii_sell': float(entry.get('fiiSellValue', 0) or 0),
                            'dii_buy':  float(entry.get('diiBuyValue', 0) or 0),
                            'dii_sell': float(entry.get('diiSellValue', 0) or 0),
                        }
                except Exception:
                    continue

            if not by_date:
                return None

            df = pd.DataFrame.from_dict(by_date, orient='index').sort_index()
            df.index.name = 'date'

            # The API only returns the last few sessions; the 5d rolling
            # signal needs history. Accumulate: merge with the disk cache,
            # new data winning on overlap. Drop all-zero legacy rows (the
            # old parser bug wrote zeros — they would poison the rolling sum).
            base_cols = ['fii_buy', 'fii_sell', 'dii_buy', 'dii_sell']
            if cls._cache_file.exists():
                try:
                    old = pd.read_parquet(cls._cache_file)
                    old = old[[c for c in base_cols if c in old.columns]]
                    old = old[old.abs().sum(axis=1) > 0]
                    df = df.combine_first(old).sort_index()
                except Exception:
                    pass
            df['fii_net'] = df['fii_buy'] - df['fii_sell']
            df['dii_net'] = df['dii_buy'] - df['dii_sell']
            df['combined_net'] = df['fii_net'] + df['dii_net']

            # Rolling 5-day momentum
            df['fii_net_5d'] = df['fii_net'].rolling(5).sum()
            df['combined_net_5d'] = df['combined_net'].rolling(5).sum()

            # Cache
            try:
                df.to_parquet(cls._cache_file)
            except Exception:
                pass

            cls._cached_df = df
            cls._cache_date = today
            return df

        except Exception as e:
            print(f"   [WARN] FII/DII fetch error: {e}")
            return None

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> Tuple[float, float]:
        """
        Returns (score [-1,1], confidence [0,100]).

        Score > 0 = bullish (strong FII net buy), Score < 0 = bearish.
        This is a market-wide signal — same for all Nifty 50 stocks.
        For small/midcap stocks, dampen by 0.5x.
        """
        df = cls.fetch_flows()
        if df is None or len(df) == 0:
            return 0.0, 0.0

        # Get most recent data up to as_of_date
        past = df[df.index.date <= as_of_date]
        if len(past) < 5:
            return 0.0, 0.0

        latest = past.iloc[-1]
        fii_5d = latest.get('fii_net_5d', 0) or 0
        combined_5d = latest.get('combined_net_5d', 0) or 0

        # Normalize: ±5000 crore = ±1.0 score
        score = np.clip(combined_5d / 5000.0, -1.0, 1.0)
        confidence = min(abs(score) * 100, 90.0)

        return float(score), float(confidence)


# ==============================================================================
# 2. BULK DEAL FOLLOW-THROUGH ALPHA
# ==============================================================================
class BulkDealAlpha:
    """
    Bulk/block deal follow-through.

    NSE publishes bulk deals (>0.5% of equity traded by one client in
    one session) and block deals. Research shows follow-through buying
    persists for 3-10 days after large operator accumulation.

    Data: NSE bulk deals CSV (free, daily)
    Edge: Identify informed accumulation before price discovery.
    """

    _cache_file = _CACHE_DIR / "bulk_deals.parquet"
    # Static archive CSV (works from any IP incl. GitHub runners; no cookies).
    _CSV_URL = "https://nsearchives.nseindia.com/content/equities/bulk.csv"

    @classmethod
    def fetch_bulk_deals(cls, days_back: int = 30) -> Optional[pd.DataFrame]:
        """
        Fetch bulk deals from NSE's daily archive CSV and merge with cached
        history. The CSV holds the latest session's deals, so we accumulate it
        in a parquet cache to build the multi-day lookback window over time.
        """
        # 1. Load cached history
        hist = None
        if cls._cache_file.exists():
            try:
                hist = pd.read_parquet(cls._cache_file)
            except Exception:
                hist = None

        # 2. Download today's archive CSV
        new_df = None
        try:
            resp = _safe_get(cls._CSV_URL,
                             headers={'Referer': 'https://www.nseindia.com/'})
            if resp is not None and resp.text and ',' in resp.text:
                raw = pd.read_csv(io.StringIO(resp.text))
                raw.columns = [c.strip() for c in raw.columns]
                colmap = {}
                for c in raw.columns:
                    lc = c.lower()
                    if lc.startswith('date'):                colmap[c] = 'date'
                    elif lc == 'symbol':                     colmap[c] = 'symbol'
                    elif 'client' in lc:                     colmap[c] = 'client'
                    elif 'buy' in lc and 'sell' in lc:       colmap[c] = 'buy_sell'
                    elif 'quantity' in lc:                   colmap[c] = 'quantity'
                    elif 'price' in lc and 'price' not in colmap.values():
                        colmap[c] = 'price'
                raw = raw.rename(columns=colmap)
                req = ['date', 'symbol', 'buy_sell', 'quantity']
                if all(k in raw.columns for k in req):
                    keep = [k for k in ['date', 'symbol', 'client', 'buy_sell',
                                        'quantity', 'price'] if k in raw.columns]
                    raw = raw[keep].copy()
                    raw['date'] = pd.to_datetime(raw['date'], dayfirst=True,
                                                 errors='coerce')
                    raw['symbol'] = raw['symbol'].astype(str).str.strip().str.upper()
                    # Normalise BUY/SELL -> B/S
                    raw['buy_sell'] = (raw['buy_sell'].astype(str).str.strip()
                                       .str.upper().str[0])
                    raw['quantity'] = pd.to_numeric(
                        raw['quantity'].astype(str).str.replace(',', '', regex=False),
                        errors='coerce')
                    if 'price' in raw.columns:
                        raw['price'] = pd.to_numeric(
                            raw['price'].astype(str).str.replace(',', '', regex=False),
                            errors='coerce')
                    if 'client' not in raw.columns:
                        raw['client'] = ''
                    raw = raw.dropna(subset=['date', 'symbol', 'quantity'])
                    if not raw.empty:
                        new_df = raw.set_index('date').sort_index()
        except Exception as e:
            print(f"   [WARN] Bulk deals fetch error: {e}")
            new_df = None

        # 3. Merge history + new, dedupe, keep last 60 days
        frames = [f for f in [hist, new_df] if f is not None and not f.empty]
        if not frames:
            return None
        combined = pd.concat(frames).reset_index()
        dedup_cols = [c for c in ['date', 'symbol', 'client', 'buy_sell', 'quantity']
                      if c in combined.columns]
        combined = combined.drop_duplicates(subset=dedup_cols)
        combined = combined.set_index('date').sort_index()
        cutoff = pd.Timestamp(date.today() - timedelta(days=60))
        combined = combined[combined.index >= cutoff]
        try:
            combined.to_parquet(cls._cache_file)
        except Exception:
            pass
        return combined

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   lookback_days: int = 5) -> Tuple[float, float]:
        """
        Returns (score, confidence).

        Checks if significant BUY bulk deals happened in last `lookback_days`.
        Score = net_buy_qty / total_qty in that window.
        """
        df = cls.fetch_bulk_deals(days_back=30)
        if df is None:
            return 0.0, 0.0

        nse_sym = symbol.replace('.NS', '').upper()
        cutoff = as_of_date - timedelta(days=lookback_days)

        mask = (
            (df['symbol'] == nse_sym) &
            (df.index.date >= cutoff) &
            (df.index.date <= as_of_date)
        )
        recent = df[mask]

        if len(recent) == 0:
            return 0.0, 0.0

        buy_qty = recent[recent['buy_sell'] == 'B']['quantity'].sum()
        sell_qty = recent[recent['buy_sell'] == 'S']['quantity'].sum()
        total_qty = buy_qty + sell_qty

        if total_qty == 0:
            return 0.0, 0.0

        # Net buy ratio: +1 = all buys, -1 = all sells
        score = (buy_qty - sell_qty) / total_qty
        confidence = min(len(recent) * 15, 80.0)  # More deals = more confident

        return float(np.clip(score, -1.0, 1.0)), float(confidence)


# ==============================================================================
# 3. INSIDER BUYING ALPHA
# ==============================================================================
class InsiderAlpha:
    """
    Insider buying and selling (SEBI insider-trading disclosures).

    Promoters, directors and key managers must report every trade within
    two days. Net open-market buying by them over the last 90 days, weighted
    towards recent trades, scores positive; net selling negative. Gifts,
    ESOPs, pledges and transfers between family members are ignored.
    Research: insider purchases, especially by promoters, predict
    outperformance over the following months.

    Data: NSE's static insider-trading RSS feed and the XBRL filing behind
    each item, collected into a history by nse_disclosures.py (the old BSE
    and NSE JSON endpoints are blocked from GitHub's servers).
    """

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> Tuple[float, float]:
        from nse_disclosures import score
        return score('insider', symbol, as_of_date)

# ==============================================================================
# 4. POST-EARNINGS ANNOUNCEMENT DRIFT (PEAD) ALPHA
# ==============================================================================
class PEADAlpha:
    """
    Post-Earnings Announcement Drift.

    After a positive earnings surprise, stocks drift upward for 30-60 days.
    This anomaly exists in Indian markets and institutional constraint means
    it persists longer here than in the US.

    Data: Earnings dates from earnings_calendar parquet files (already cached)
    Edge: Systematic drift that retail ignores and institutions exploit slowly.

    Score logic:
      - Strong beat (>10% surprise) within 60 days → +1.0
      - Moderate beat (>5%) within 60 days → +0.6
      - Miss (< -5%) within 60 days → -0.8
      - No earnings in window → 0.0
    """

    _earnings_dir = Path("data/earnings_calendar")

    @staticmethod
    def _surprise_pct(row: pd.Series) -> Optional[float]:
        """EPS surprise in % for one earnings row.

        earnings_calendar.py caches yfinance's earnings_dates frame as-is,
        whose columns are 'EPS Estimate' / 'Reported EPS' / 'Surprise(%)'.
        This alpha used to look only for 'surprise_pct'-style names, so it
        never found a surprise and scored 0 on every stock for months.
        Reported vs estimate is computed directly (no unit ambiguity); the
        other columns are fallbacks for older caches.
        """
        rep, est = row.get('Reported EPS'), row.get('EPS Estimate')
        if (rep is not None and est is not None and not pd.isna(rep)
                and not pd.isna(est) and est != 0):
            return (float(rep) - float(est)) / abs(float(est)) * 100
        for col in ['Surprise(%)', 'surprise_pct', 'eps_surprise_pct',
                    'EPS_Surprise_Pct', 'Surprise_Pct', 'beat_pct']:
            v = row.get(col)
            if v is not None and not pd.isna(v):
                return float(v)
        for col in ['beat_estimates', 'beat', 'Beat']:
            v = row.get(col)
            if v is not None and not pd.isna(v):
                return 8.0 if v else -8.0
        return None

    @classmethod
    def _load_earnings(cls, symbol: str) -> Optional[pd.DataFrame]:
        """Load cached earnings data for symbol."""
        nse_sym = symbol.replace('.NS', '').upper()
        for fname in [f"{nse_sym}_earnings.parquet", f"{symbol}_earnings.parquet"]:
            fpath = cls._earnings_dir / fname
            if fpath.exists():
                try:
                    return pd.read_parquet(fpath)
                except Exception:
                    pass
        return None

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   drift_window_days: int = 60) -> Tuple[float, float]:
        """
        Returns (score, confidence).

        Checks if there was an earnings release in the last `drift_window_days`
        and whether it was a beat or miss.
        """
        df = cls._load_earnings(symbol)
        if df is None or df.empty:
            return 0.0, 0.0

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        cutoff = as_of_date - timedelta(days=drift_window_days)
        recent = df[(df.index.date >= cutoff) & (df.index.date <= as_of_date)]

        if recent.empty:
            return 0.0, 0.0

        # Most recent earnings event whose surprise can be measured
        surprise, days_since = None, None
        for when, row in recent.iloc[::-1].iterrows():
            surprise = cls._surprise_pct(row)
            if surprise is not None:
                days_since = (as_of_date - when.date()).days
                break

        if surprise is None:
            return 0.0, 0.0

        # Score: decay over drift window
        time_decay = max(0.0, 1.0 - days_since / drift_window_days)

        if surprise > 10:
            base_score = 1.0
        elif surprise > 5:
            base_score = 0.6
        elif surprise > 0:
            base_score = 0.3
        elif surprise < -10:
            base_score = -0.8
        elif surprise < -5:
            base_score = -0.5
        else:
            base_score = -0.2

        score = base_score * time_decay
        confidence = min(abs(surprise) * 5, 80.0) * time_decay

        return float(score), float(confidence)


# ==============================================================================
# 5. F&O BAN LIST MEAN-REVERSION ALPHA
# ==============================================================================
class FOBanAlpha:
    """
    F&O Ban List Mean-Reversion.

    NSE bans stocks from new F&O positions when market-wide position limit
    (MWPL) exceeds 95%. These stocks are often overcrowded with shorts.
    When they EXIT the ban list, shorts must cover → abnormal positive returns
    in next 2-5 days.

    Data: NSE ban list (free, daily PDF/HTML)
    Edge: Mechanical short-covering pressure that algorithms exploit late.
    """

    _cache_file = _CACHE_DIR / "fo_ban_list.json"
    _yesterday_ban: set = set()
    _today_ban: set = set()
    _last_fetch: Optional[date] = None

    @classmethod
    def fetch_ban_list(cls, as_of_date: Optional[date] = None) -> set:
        """Fetch current F&O ban list from NSE."""
        today = as_of_date or date.today()

        if cls._last_fetch == today and cls._today_ban:
            return cls._today_ban

        import re
        ban_symbols: set = set()

        # 1. Static archive CSV (works from any IP incl. GitHub; no cookies).
        #    Format: a header line + numbered rows like "1,AMBER".
        try:
            csv_url = "https://nsearchives.nseindia.com/content/fo/fo_secban.csv"
            resp = _safe_get(csv_url, headers={'Referer': 'https://www.nseindia.com/'})
            if resp is not None and resp.text:
                for sym in re.findall(r'\d+\s*,\s*([A-Z][A-Z0-9&\-]+)', resp.text):
                    ban_symbols.add(sym.strip().upper())
        except Exception:
            pass

        # 2. Fallback: JSON API (needs cookies; may be empty)
        if not ban_symbols:
            url = "https://www.nseindia.com/api/snapshot-derivatives-equity?index=ban_list"
            resp = _safe_get(url, headers={'Referer': 'https://www.nseindia.com/'})
            if resp:
                try:
                    data = resp.json()
                    entries = data.get('data', data) if isinstance(data, dict) else data
                    for entry in entries or []:
                        sym = (entry.get('symbol', '') or entry.get('SYMBOL', '')) \
                            if isinstance(entry, dict) else str(entry)
                        if sym:
                            ban_symbols.add(str(sym).strip().upper())
                except Exception:
                    pass

        if ban_symbols:
            try:
                with open(cls._cache_file, 'w') as f:
                    import json
                    json.dump({'date': str(today), 'symbols': list(ban_symbols)}, f)
            except Exception:
                pass
            cls._last_fetch = today
            cls._today_ban = ban_symbols
            return ban_symbols

        # Disk cache fallback
        if cls._cache_file.exists():
            try:
                import json
                with open(cls._cache_file) as f:
                    cached = json.load(f)
                return set(cached.get('symbols', []))
            except Exception:
                pass

        return set()

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> Tuple[float, float]:
        """
        Returns (score, confidence).

        +0.8 if stock JUST EXITED ban list (short covering opportunity)
        -0.3 if stock is NEWLY ENTERING ban list (overcrowded, avoid)
        0.0 if no ban activity
        """
        nse_sym = symbol.replace('.NS', '').upper()

        today_ban = cls.fetch_ban_list(as_of_date)
        yesterday_ban = cls.fetch_ban_list(as_of_date - timedelta(days=1))

        # Just exited ban → mean-reversion BUY
        if nse_sym in yesterday_ban and nse_sym not in today_ban:
            return 0.8, 75.0

        # Newly entered ban → caution (overcrowded longs/shorts, momentum fading)
        if nse_sym not in yesterday_ban and nse_sym in today_ban:
            return -0.3, 50.0

        # Still in ban → slight negative (congested)
        if nse_sym in today_ban:
            return -0.2, 40.0

        return 0.0, 0.0


# ==============================================================================
# 6. CROSS-SECTIONAL MOMENTUM ALPHA
# ==============================================================================
class MomentumAlpha:
    """
    12-1 Month Cross-Sectional Momentum (Jegadeesh-Titman).

    Buy top quintile (best 20% performers over past 12m excluding last 1m).
    Sell bottom quintile. Classic factor with academic backing.

    Edge: Persistent in India; most retail ignores long-horizon momentum.
    Data: Cached OHLCV parquet files (already available)
    """

    @staticmethod
    def get_signal(symbol: str, as_of_date: date,
                   data_dir: str = "data/stocks",
                   universe_returns: Optional[Dict[str, float]] = None) -> Tuple[float, float]:
        """
        Returns (score, confidence).

        score = percentile rank in universe momentum (normalised to [-1, 1])
        If universe_returns not provided, computes stock's own momentum
        and maps to score using fixed thresholds.
        """
        nse_sym = symbol.replace('.NS', '').upper()
        fpath = Path(data_dir) / f"{nse_sym}.parquet"

        if not fpath.exists():
            return 0.0, 0.0

        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()

            cutoff = pd.Timestamp(as_of_date)
            df = df[df.index <= cutoff]

            if len(df) < 252:
                return 0.0, 0.0

            # 12-1 month return: skip last 21 days, use 252→22 range
            ret_12_1 = (df['Close'].iloc[-252] > 0 and
                        df['Close'].iloc[-22] / df['Close'].iloc[-252] - 1)
            if not isinstance(ret_12_1, float):
                return 0.0, 0.0

            # Without universe context, use absolute return thresholds
            if universe_returns:
                all_rets = np.array(list(universe_returns.values()))
                stock_ret = universe_returns.get(nse_sym, ret_12_1)
                pct_rank = (stock_ret > all_rets).mean()  # 0=worst, 1=best
                score = (pct_rank - 0.5) * 2  # normalise to [-1, 1]
                confidence = 70.0
            else:
                # Standalone score
                if ret_12_1 > 0.30:
                    score = 0.9
                elif ret_12_1 > 0.15:
                    score = 0.6
                elif ret_12_1 > 0.05:
                    score = 0.3
                elif ret_12_1 < -0.20:
                    score = -0.8
                elif ret_12_1 < -0.10:
                    score = -0.5
                else:
                    score = 0.0
                confidence = 60.0

            return float(np.clip(score, -1.0, 1.0)), float(confidence)

        except Exception as e:
            print(f"   [WARN] MomentumAlpha error for {symbol}: {e}")
            return 0.0, 0.0


# ==============================================================================
# 7. RSI MEAN REVERSION ALPHA
# ==============================================================================
class MeanRevAlpha:
    """
    5-day RSI Mean Reversion for index stocks.

    When RSI(5) < 20 on an index constituent, the stock is extremely
    oversold. These positions recover within 3-5 days ~70% of the time
    due to institutional rebalancing.

    Edge: High win rate for 2-3 day holding period.
    """

    @staticmethod
    def get_signal(symbol: str, as_of_date: date,
                   data_dir: str = "data/stocks") -> Tuple[float, float]:
        """
        Returns (score, confidence).

        RSI(5) < 20 → strong BUY (+0.9)
        RSI(5) 20-30 → moderate BUY (+0.5)
        RSI(5) > 80 → strong SELL (-0.9)
        RSI(5) 70-80 → moderate SELL (-0.5)
        """
        nse_sym = symbol.replace('.NS', '').upper()
        fpath = Path(data_dir) / f"{nse_sym}.parquet"

        if not fpath.exists():
            return 0.0, 0.0

        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            df = df[df.index.date <= as_of_date].sort_index()

            if len(df) < 10:
                return 0.0, 0.0

            close = df['Close']

            # Compute RSI(5)
            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(5).mean()
            rs = gain / (loss + 1e-9)
            rsi5 = 100 - (100 / (1 + rs))

            latest_rsi = rsi5.iloc[-1]
            if pd.isna(latest_rsi):
                return 0.0, 0.0

            if latest_rsi < 20:
                return 0.9, 80.0
            elif latest_rsi < 30:
                return 0.5, 65.0
            elif latest_rsi > 80:
                return -0.9, 80.0
            elif latest_rsi > 70:
                return -0.5, 65.0
            else:
                return 0.0, 0.0

        except Exception as e:
            print(f"   [WARN] MeanRevAlpha error for {symbol}: {e}")
            return 0.0, 0.0


# ==============================================================================
# MASTER ALPHA AGGREGATOR
# ==============================================================================
class IndiaAlphaAggregator:
    """
    Combines all India-specific alpha signals into a single composite score.

    Weights are based on historical Sharpe ratios of each alpha:
      PEAD:        0.20  (strong academic backing in India)
      Momentum:    0.18  (persistent cross-sectional edge)
      FII/DII:     0.18  (market-wide flows predict direction)
      Mean Rev:    0.15  (high win rate on short horizon)
      Bulk Deal:   0.12  (informed accumulation signal)
      Insider:     0.10  (high conviction but noisy)
      F&O Ban:     0.07  (mechanical, high precision, low frequency)

    Total = 1.00
    """

    WEIGHTS = {
        'pead':       0.20,
        'momentum':   0.18,
        'fii_dii':    0.18,
        'mean_rev':   0.15,
        'bulk_deal':  0.12,
        'insider':    0.10,
        'fo_ban':     0.07,
    }

    @classmethod
    def get_composite_signal(
        cls,
        symbol: str,
        as_of_date: date,
        data_dir: str = "data/stocks",
        universe_returns: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Compute all alpha signals and return composite.

        Returns dict with:
          score       : composite score [-1, 1]
          confidence  : composite confidence [0, 100]
          signal      : 'BUY' | 'SELL' | 'HOLD'
          components  : dict of individual alpha scores
        """
        components = {}
        weighted_score = 0.0
        total_weight_used = 0.0

        # ── Run each alpha ──────────────────────────────────────────────────
        alpha_funcs = {
            'pead':      lambda: PEADAlpha.get_signal(symbol, as_of_date),
            'momentum':  lambda: MomentumAlpha.get_signal(symbol, as_of_date, data_dir, universe_returns),
            'fii_dii':   lambda: FIIDIIAlpha.get_signal(symbol, as_of_date),
            'mean_rev':  lambda: MeanRevAlpha.get_signal(symbol, as_of_date, data_dir),
            'bulk_deal': lambda: BulkDealAlpha.get_signal(symbol, as_of_date),
            'insider':   lambda: InsiderAlpha.get_signal(symbol, as_of_date),
            'fo_ban':    lambda: FOBanAlpha.get_signal(symbol, as_of_date),
        }

        for name, func in alpha_funcs.items():
            try:
                score, conf = func()
                components[name] = {'score': round(score, 3), 'confidence': round(conf, 1)}
                if conf > 0:  # Only include if alpha has a view
                    weighted_score += score * cls.WEIGHTS[name]
                    total_weight_used += cls.WEIGHTS[name]
            except Exception as e:
                components[name] = {'score': 0.0, 'confidence': 0.0, 'error': str(e)}

        # Normalise by weights used (graceful degradation)
        if total_weight_used > 0:
            composite_score = weighted_score / total_weight_used
        else:
            composite_score = 0.0

        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        # Composite confidence = average of non-zero confidences
        confs = [v['confidence'] for v in components.values() if v['confidence'] > 0]
        composite_conf = float(np.mean(confs)) if confs else 0.0

        # Signal threshold: ±0.25 for action
        if composite_score > 0.25:
            signal = 'BUY'
        elif composite_score < -0.25:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        result = {
            'symbol': symbol,
            'date': as_of_date.isoformat(),
            'score': round(composite_score, 3),
            'confidence': round(composite_conf, 1),
            'signal': signal,
            'components': components
        }

        if verbose:
            print(f"\nIndia Alpha Signals: {symbol} ({as_of_date})")
            print(f"   Composite: {signal} | Score: {composite_score:+.3f} | Conf: {composite_conf:.0f}%")
            for name, v in components.items():
                bar = '█' * int(abs(v['score']) * 10)
                direction = '+' if v['score'] >= 0 else '-'
                print(f"   {name:10s}: {direction}{bar:<10} {v['score']:+.2f} ({v['confidence']:.0f}%)")

        return result


# ==============================================================================
# 8. DELIVERY PERCENT ALPHA  (NSE Bhav Copy)
# ==============================================================================
class DeliveryPercentAlpha:
    """
    Delivery-to-Traded Volume Percentage Signal.

    NSE publishes daily delivery % in the bhav copy:
      - High delivery % (>80%) = investors are HOLDING, not trading → bullish
      - Rising delivery % over 3 days = institutional accumulation in progress
      - Very low delivery % (<20%) = pure speculation, no conviction → avoid
      - Sudden drop in delivery % = distribution (smart money selling to traders)

    Edge: Institutional accumulation shows up as high+rising delivery % BEFORE
    the price moves. Retailers trade intraday; institutions take delivery.
    Data: NSE archives (free, daily CSV/ZIP)
    """

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> tuple:
        """Returns (score, confidence)."""
        try:
            from nse_fetcher import get_fetcher
            fetcher = get_fetcher()

            # Fetch today's bhav and 3 days ago (for trend)
            bhav_today = fetcher.get_delivery_bhav(as_of_date)
            bhav_3d    = fetcher.get_delivery_bhav(as_of_date - timedelta(days=4))

            if bhav_today is None or bhav_today.empty:
                return 0.0, 0.0

            nse_sym = symbol.replace('.NS', '').upper()
            row = bhav_today[bhav_today['SYMBOL'] == nse_sym]
            if row.empty:
                return 0.0, 0.0

            deliv_today = float(row['DELIV_PER'].iloc[0])

            # Trend: compare to 3 days ago
            deliv_trend = 0.0
            if bhav_3d is not None and not bhav_3d.empty:
                row3 = bhav_3d[bhav_3d['SYMBOL'] == nse_sym]
                if not row3.empty:
                    deliv_3d    = float(row3['DELIV_PER'].iloc[0])
                    deliv_trend = deliv_today - deliv_3d  # positive = rising

            # Score based on absolute delivery % and trend
            if deliv_today >= 80:
                score = 0.8
                conf  = 72.0
            elif deliv_today >= 65:
                score = 0.5
                conf  = 60.0
            elif deliv_today >= 50:
                score = 0.2
                conf  = 45.0
            elif deliv_today <= 20:
                score = -0.6
                conf  = 65.0
            elif deliv_today <= 35:
                score = -0.3
                conf  = 50.0
            else:
                score = 0.0
                conf  = 30.0

            # Trend bonus/penalty (±0.15 max)
            trend_boost = np.clip(deliv_trend / 100.0, -0.15, 0.15)
            score = float(np.clip(score + trend_boost, -1.0, 1.0))
            if conf > 0:
                conf = min(conf + abs(deliv_trend) * 0.5, 85.0)

            return score, conf

        except Exception as e:
            return 0.0, 0.0


# ==============================================================================
# 9. OPTION CHAIN ALPHA  (PCR + Max Pain)
# ==============================================================================
class OptionChainAlpha:
    """
    Option Chain Signal: Put-Call Ratio (PCR) + Max Pain.

    PCR Logic (contrarian):
      - PCR > 1.4  : extreme fear → contrarian BUY (+0.7)
      - PCR > 1.2  : elevated puts → mild bullish (+0.4)
      - PCR < 0.7  : extreme greed/calls → contrarian SELL (-0.6)
      - PCR < 0.9  : mild call skew → mild bearish (-0.3)

    Max Pain Logic:
      - Price significantly below max pain → stock likely to drift UP
        (market makers push toward max pain to let options expire worthless)
      - Price significantly above max pain → stock likely to drift DOWN

    Data: NSE option chain API (free, requires session cookies)
    Edge: PCR extremes predict 3-5 day reversals. Max Pain strong near expiry.
    """

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> tuple:
        """Returns (score, confidence)."""
        try:
            from nse_fetcher import get_fetcher
            fetcher = get_fetcher()

            oc = fetcher.get_option_chain(symbol)
            if not oc:
                return 0.0, 0.0

            pcr        = oc.get('pcr', 1.0)
            max_pain   = oc.get('max_pain', 0.0)
            underlying = oc.get('underlying', 0.0)

            # PCR signal (contrarian)
            if pcr >= 1.5:
                pcr_score = 0.75
                pcr_conf  = 75.0
            elif pcr >= 1.2:
                pcr_score = 0.45
                pcr_conf  = 60.0
            elif pcr >= 0.9:
                pcr_score = 0.0
                pcr_conf  = 0.0
            elif pcr >= 0.7:
                pcr_score = -0.35
                pcr_conf  = 55.0
            else:
                pcr_score = -0.65
                pcr_conf  = 72.0

            # Max pain signal
            mp_score = 0.0
            mp_conf  = 0.0
            if underlying > 0 and max_pain > 0:
                gap_pct = (max_pain - underlying) / underlying
                if abs(gap_pct) > 0.01:   # >1% gap = meaningful
                    if gap_pct > 0.04:    # price 4%+ below max pain → drift up
                        mp_score = 0.5
                        mp_conf  = 60.0
                    elif gap_pct > 0.02:
                        mp_score = 0.3
                        mp_conf  = 45.0
                    elif gap_pct < -0.04: # price 4%+ above max pain → drift down
                        mp_score = -0.5
                        mp_conf  = 60.0
                    elif gap_pct < -0.02:
                        mp_score = -0.3
                        mp_conf  = 45.0

            # Combine PCR + Max Pain (60/40 split)
            if pcr_conf > 0 and mp_conf > 0:
                final_score = pcr_score * 0.60 + mp_score * 0.40
                final_conf  = (pcr_conf * 0.60 + mp_conf * 0.40)
            elif pcr_conf > 0:
                final_score = pcr_score
                final_conf  = pcr_conf
            elif mp_conf > 0:
                final_score = mp_score
                final_conf  = mp_conf
            else:
                return 0.0, 0.0

            return float(np.clip(final_score, -1.0, 1.0)), float(final_conf)

        except Exception as e:
            return 0.0, 0.0


# ==============================================================================
# 10. CORPORATE EVENT ALPHA  (NSE Announcements)
# ==============================================================================
class CorporateEventAlpha:
    """
    Corporate Event Signal from NSE announcements.

    Announcement type → score:
      BUYBACK         : +0.8  (strong bullish — company buying own stock)
      DIVIDEND        : +0.4  (bullish — cash to shareholders)
      BONUS           : +0.3  (neutral-bullish — signals confidence)
      ACQUISITION     : +0.3  (can be transformative)
      FINANCIAL RESULTS: contextual (score by profit growth in text)
      RIGHTS ISSUE    : -0.4  (dilution — bearish)
      QIP / FPO       : -0.3  (dilution)
      DEBT / LOAN     : -0.2  (leverage concern)
      PLEDGE          : -0.6  (promoter pledging shares — red flag)
      RESIGNATION     : -0.3  (management instability)

    Recency decay: events older than 7 days have reduced weight.
    Data: NSE corporate announcements API (free).
    """

    # Keywords → (score, confidence)
    _PATTERNS: List[tuple] = [
        # Very bullish
        ('buyback',           0.80, 78.0),
        ('buy-back',          0.80, 78.0),
        ('share repurchase',  0.75, 75.0),
        # Bullish
        ('dividend',          0.40, 62.0),
        ('bonus issue',       0.35, 58.0),
        ('bonus shares',      0.35, 58.0),
        ('acquisition',       0.30, 50.0),
        ('merger',            0.25, 45.0),
        ('demerger',          0.20, 40.0),
        ('capacity expansion',0.25, 50.0),
        ('order received',    0.30, 55.0),
        ('new order',         0.30, 55.0),
        ('contract awarded',  0.30, 55.0),
        # Bearish
        ('rights issue',     -0.40, 65.0),
        ('rights entitlement',-0.35, 60.0),
        ('qip',              -0.30, 58.0),
        ('preferential allotment', -0.25, 52.0),
        ('fpo',              -0.25, 52.0),
        ('pledge',           -0.55, 70.0),
        ('pledged',          -0.55, 70.0),
        ('resignation',      -0.30, 55.0),
        ('regulatory action',-0.40, 65.0),
        ('sebi notice',      -0.45, 70.0),
        ('default',          -0.60, 72.0),
        ('npa',              -0.40, 65.0),
        ('fraud',            -0.70, 80.0),
        ('insolvency',       -0.70, 80.0),
        # Extended lexicon
        ('open offer',        0.45, 62.0),
        ('delisting',         0.40, 58.0),
        ('rating upgrade',    0.40, 60.0),
        ('upgraded to',       0.35, 55.0),
        ('joint venture',     0.25, 48.0),
        ('partnership',       0.20, 45.0),
        ('record date',       0.15, 35.0),
        ('stake acquisition', 0.30, 52.0),
        ('preferential issue',-0.25, 52.0),
        ('rating downgrade', -0.45, 65.0),
        ('downgraded to',    -0.40, 60.0),
        ('credit watch',     -0.35, 58.0),
        ('penalty',          -0.30, 55.0),
        ('show cause',       -0.40, 62.0),
        ('income tax search',-0.45, 65.0),
        ('gst notice',       -0.25, 50.0),
        ('litigation',       -0.20, 45.0),
        ('arbitration',      -0.20, 45.0),
        ('plant shutdown',   -0.35, 58.0),
        ('fire at',          -0.30, 55.0),
        ('strike at',        -0.25, 50.0),
        ('ceo steps down',   -0.35, 58.0),
        ('cfo steps down',   -0.40, 60.0),
        ('auditor resign',   -0.65, 78.0),
    ]

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> tuple:
        """Returns (score, confidence)."""
        try:
            from nse_fetcher import get_fetcher
            fetcher = get_fetcher()

            announcements = fetcher.get_announcements(symbol)
            if not announcements:
                return 0.0, 0.0

            # Filter to last 30 days
            scores = []
            for ann in announcements:
                ann_date_str = ann.get('an_date', '') or ann.get('date', '')
                if not ann_date_str:
                    continue

                # Parse date
                ann_date = None
                for fmt in ('%d-%b-%Y %H:%M:%S', '%d-%b-%Y', '%Y-%m-%d',
                            '%d/%m/%Y', '%d %b %Y'):
                    try:
                        ann_date = datetime.strptime(
                            ann_date_str.strip()[:20], fmt
                        ).date()
                        break
                    except Exception:
                        continue

                if ann_date is None:
                    continue

                days_old = (as_of_date - ann_date).days
                if days_old < 0 or days_old > 30:
                    continue

                # Recency decay: events decay to 30% weight over 30 days
                recency = max(0.3, 1.0 - days_old / 35.0)

                text = (ann.get('desc', '') + ' ' +
                        ann.get('attachment', '')).lower()

                for keyword, raw_score, raw_conf in cls._PATTERNS:
                    if keyword in text:
                        scores.append((raw_score * recency, raw_conf * recency))
                        break  # one match per announcement

            if not scores:
                return 0.0, 0.0

            # Take strongest signal (abs value)
            scores.sort(key=lambda x: abs(x[0]), reverse=True)
            best_score, best_conf = scores[0]

            # If multiple signals in same direction, slight boost
            if len(scores) > 1:
                same_dir = sum(1 for s, _ in scores[1:] if s * best_score > 0)
                if same_dir > 0:
                    best_score *= min(1.15, 1 + same_dir * 0.05)
                    best_conf  = min(best_conf * 1.1, 85.0)

            return float(np.clip(best_score, -1.0, 1.0)), float(best_conf)

        except Exception as e:
            return 0.0, 0.0


# ==============================================================================
# 11. RELATIVE STRENGTH ALPHA  (stock vs Nifty 50)
# ==============================================================================
class RelativeStrengthAlpha:
    """
    Relative strength: is the stock LEADING or LAGGING the index?

    Even in a flat market, leaders keep leading and laggards keep lagging
    (cross-sectional persistence). We measure the stock's 63-day (~3 month)
    return minus the Nifty 50's, so it is largely independent of absolute
    momentum (a stock can be up but still underperform the index).
    """

    _index_cache: dict = {}

    @classmethod
    def _load_close(cls, fname: str, data_dir: str):
        fpath = Path(data_dir) / f"{fname}.parquet"
        if not fpath.exists():
            return None
        try:
            df = pd.read_parquet(fpath)
            df.index = pd.to_datetime(df.index)
            return df.sort_index()['Close']
        except Exception:
            return None

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   data_dir: str = "data/stocks") -> Tuple[float, float]:
        nse = symbol.replace('.NS', '').replace('^', '').upper()
        close = cls._load_close(nse, data_dir)
        idx = cls._load_close("NIFTY50", data_dir)
        if close is None or idx is None or len(close) < 70 or len(idx) < 70:
            return 0.0, 0.0
        cutoff = pd.Timestamp(as_of_date)
        close = close[close.index <= cutoff]
        idx = idx[idx.index <= cutoff]
        if len(close) < 65 or len(idx) < 65:
            return 0.0, 0.0
        try:
            stock_ret = float(close.iloc[-1] / close.iloc[-63] - 1)
            index_ret = float(idx.iloc[-1] / idx.iloc[-63] - 1)
        except Exception:
            return 0.0, 0.0
        rs = stock_ret - index_ret  # relative outperformance
        # Map ±12% relative move to ±0.9 score
        score = float(np.clip(rs / 0.12 * 0.9, -0.9, 0.9))
        conf = float(np.clip(40 + abs(rs) * 300, 0, 75)) if abs(score) > 0.1 else 0.0
        return score, conf


class Hi52Alpha:
    """
    52-week-high proximity (George & Hwang 2004).

    Stocks trading near their 52-week high keep outperforming: investors
    anchor on the high and are slow to bid past it, so good news gets
    underreacted to near the high. One of the most robust cross-sectional
    anomalies, and it replicates in India. Confirmed in our own panel:
    hi52_dist is a top ML feature and momentum 12-1 (its cousin) has
    IC +0.026 (walk_forward_report.md).

    Needs only the price parquets — works identically in the cloud.
    """

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   data_dir: str = "data/stocks") -> Tuple[float, float]:
        nse = symbol.replace('.NS', '').replace('^', '').upper()
        close = RelativeStrengthAlpha._load_close(nse, data_dir)
        if close is None:
            return 0.0, 0.0
        close = close[close.index <= pd.Timestamp(as_of_date)]
        if len(close) < 252:
            return 0.0, 0.0
        try:
            ratio = float(close.iloc[-1] / close.iloc[-252:].max())
        except Exception:
            return 0.0, 0.0
        # ratio 1.0 = at the high; <0.7 = deep below (loser territory)
        if ratio >= 0.97:
            return 0.8, 70.0          # at/near high — strongest continuation
        if ratio >= 0.90:
            return 0.5, 60.0
        if ratio >= 0.80:
            return 0.2, 45.0
        if ratio <= 0.60:
            return -0.4, 55.0         # deep losers keep lagging at 5-20d horizon
        if ratio <= 0.70:
            return -0.2, 45.0
        return 0.0, 0.0               # middle zone — no edge


# ==============================================================================
# 12. SECTOR ROTATION ALPHA  (which sectors are in favour)
# ==============================================================================
class SectorRotationAlpha:
    """
    Sector rotation: capital rotates between sectors. A fundamentally average
    stock in a hot sector often beats a great stock in a cold sector.

    We rank every sector by its members' median 21-day return, then score a
    stock by where its sector sits in that ranking. This is genuinely
    independent of single-stock momentum/mean-reversion.
    """

    _cache: dict = {}   # keyed by (as_of_date, data_dir) -> {sector: zscore}

    @classmethod
    def _sector_strength(cls, as_of_date: date, data_dir: str) -> dict:
        key = (str(as_of_date), data_dir)
        if key in cls._cache:
            return cls._cache[key]
        try:
            from sector_mapping import STOCK_SECTOR_MAP
        except Exception:
            cls._cache[key] = {}
            return {}

        cutoff = pd.Timestamp(as_of_date)
        sector_rets: Dict[str, list] = {}
        for sym, sector in STOCK_SECTOR_MAP.items():
            nse = sym.replace('.NS', '').upper()
            fpath = Path(data_dir) / f"{nse}.parquet"
            if not fpath.exists():
                continue
            try:
                c = pd.read_parquet(fpath)
                c.index = pd.to_datetime(c.index)
                c = c.sort_index()['Close']
                c = c[c.index <= cutoff]
                if len(c) < 25:
                    continue
                r = float(c.iloc[-1] / c.iloc[-21] - 1)
                sector_rets.setdefault(sector, []).append(r)
            except Exception:
                continue

        # median return per sector, then z-score across sectors
        med = {s: float(np.median(v)) for s, v in sector_rets.items() if len(v) >= 2}
        if len(med) < 3:
            cls._cache[key] = {}
            return {}
        vals = np.array(list(med.values()))
        mu, sd = float(vals.mean()), float(vals.std())
        z = {s: ((v - mu) / sd if sd > 0 else 0.0) for s, v in med.items()}
        cls._cache[key] = z
        return z

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   data_dir: str = "data/stocks") -> Tuple[float, float]:
        try:
            from sector_mapping import STOCK_SECTOR_MAP
        except Exception:
            return 0.0, 0.0
        key = symbol if symbol.endswith('.NS') else symbol + '.NS'
        sector = STOCK_SECTOR_MAP.get(key)
        if not sector:
            return 0.0, 0.0
        z = cls._sector_strength(as_of_date, data_dir)
        if sector not in z:
            return 0.0, 0.0
        # z-score -> score: ±2 sigma maps to ±0.8
        score = float(np.clip(z[sector] / 2.0 * 0.8, -0.8, 0.8))
        conf = float(np.clip(35 + abs(z[sector]) * 20, 0, 70)) if abs(score) > 0.1 else 0.0
        return score, conf


# ==============================================================================
# 13. PROMOTER PLEDGE ALPHA  (NSE pledge filings)
# ==============================================================================
class PromoterPledgeAlpha:
    """
    Promoter share pledging.

    Promoters pledging their own shares to borrow money is a red flag in
    Indian markets: lenders sell pledged stakes on margin calls. Pledge
    creation by promoters or directors over the last 180 days scores
    negative (an invocation, when the lender actually seizes shares, counts
    double); pledge release scores positive.

    Data: the same SEBI insider-trading filings, which carry pledge events
    (nse_disclosures.py). The old NSE pledge API is blocked from GitHub's
    servers, and the SAST pledge feed only links to scanned PDFs.
    """

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> Tuple[float, float]:
        from nse_disclosures import score
        return score('pledge', symbol, as_of_date)

# ==============================================================================
# 14. SAST STAKE-CHANGE ALPHA  (Reg 29 disclosures)
# ==============================================================================
class SASTAlpha:
    """
    Substantial stake changes disclosed under SAST Regulation 29.

    When a promoter or large acquirer crosses a disclosure threshold buying
    shares in the open market, they are betting real money with the best
    information available. Promoter acquisitions predict outperformance;
    promoter sales are a warning.

    Score (events in the last 30 days, promoter events weighted double):
      net acquisition → up to +0.7    net sale → down to -0.7
    Data: NSE corporate-sast-reg29 (free).
    """

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   lookback_days: int = 30) -> Tuple[float, float]:
        try:
            from nse_fetcher import get_fetcher
            deals = get_fetcher().get_sast_deals()
            if not deals:
                return 0.0, 0.0

            nse_sym = symbol.replace('.NS', '').upper()
            cutoff = as_of_date - timedelta(days=lookback_days)

            net, n_events = 0.0, 0
            for d in deals:
                if d.get('symbol') != nse_sym:
                    continue
                try:
                    end = datetime.strptime(
                        d.get('end_date', '')[:11].strip(), '%d-%b-%Y').date()
                except Exception:
                    continue
                if end < cutoff or end > as_of_date:
                    continue

                direction = 1.0 if 'acqui' in d.get('acq_sale', '').lower() else -1.0
                # Size term: 1%+ of diluted capital is a big statement
                size = min(abs(d.get('pct_diluted', 0.0)) / 1.0, 1.0)
                weight = (2.0 if d.get('is_promoter') else 1.0) * (0.4 + 0.6 * size)
                net += direction * weight
                n_events += 1

            if n_events == 0:
                return 0.0, 0.0

            score = float(np.clip(net * 0.35, -0.7, 0.7))
            conf  = float(min(45 + n_events * 10 + abs(net) * 10, 80.0))
            if abs(score) < 0.05:
                return 0.0, 0.0
            return score, conf
        except Exception:
            return 0.0, 0.0


# ==============================================================================
# 15. SHAREHOLDING DELTA ALPHA  (quarterly promoter stake changes)
# ==============================================================================
class ShareholdingDeltaAlpha:
    """
    Quarter-over-quarter change in promoter holding from official shareholding
    patterns. Promoters raising their stake = the strongest insider signal
    available in public data; consistent dilution = caution.

    Point-in-time safe: only quarters whose SUBMISSION date (when the filing
    became public) is on or before as_of_date are used — never the quarter-end
    date, which would be lookahead.

    Data: NSE corporate-share-holdings-master per symbol (free, cached 30d).
    """

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> Tuple[float, float]:
        try:
            from nse_fetcher import get_fetcher
            rows = get_fetcher().get_shareholding(symbol)
            if not rows or len(rows) < 2:
                return 0.0, 0.0

            def _parse(d: str) -> Optional[date]:
                for fmt in ('%d-%b-%Y', '%d-%B-%Y', '%Y-%m-%d'):
                    try:
                        return datetime.strptime(d.strip()[:11], fmt).date()
                    except Exception:
                        continue
                return None

            known = []
            for r in rows:
                sub = _parse(r.get('submission_date', ''))
                qe  = _parse(r.get('quarter_end', ''))
                if sub and qe and sub <= as_of_date and r.get('promoter_pct'):
                    known.append((qe, sub, float(r['promoter_pct'])))
            if len(known) < 2:
                return 0.0, 0.0

            known.sort(key=lambda x: x[0])
            (q_prev, _, p_prev), (q_last, sub_last, p_last) = known[-2], known[-1]
            delta = p_last - p_prev

            if delta >= 0.5:
                score, conf = 0.5, 65.0
            elif delta >= 0.25:
                score, conf = 0.3, 50.0
            elif delta <= -0.5:
                score, conf = -0.5, 65.0
            elif delta <= -0.25:
                score, conf = -0.3, 50.0
            else:
                return 0.0, 0.0

            # Decay: filings older than ~a quarter carry less signal
            days_old = (as_of_date - sub_last).days
            decay = max(0.4, 1.0 - days_old / 180.0)
            return float(np.clip(score * decay, -1.0, 1.0)), float(conf * decay)
        except Exception:
            return 0.0, 0.0


# ==============================================================================
# RISK-ADJUSTED MOMENTUM ALPHA  (NIFTY200 Momentum 30 style, cross-sectional)
# ==============================================================================
class RiskAdjMomentumAlpha:
    """
    Risk-adjusted momentum: 6- and 12-month returns (to one month ago), each
    divided by the stock's 1-year volatility, standardised across the
    universe and averaged — the ranking behind NSE's NIFTY200 Momentum 30.

    Plain momentum rewards a stock that jumped 60% on one burst of news as
    much as one that climbed 60% steadily; dividing by volatility prefers the
    steady climber, which is less prone to the sharp reversals that make
    momentum crash. Definitions live in factors.py and are shared with the
    research lab, so the backtested and live versions are identical.
    """

    _key: Optional[Tuple] = None
    _scores: Dict[str, float] = {}

    @classmethod
    def _build(cls, as_of_date: date, data_dir: str) -> Dict[str, float]:
        from factors import riskadj_scores, rank_to_score
        closes = {}
        cutoff = pd.Timestamp(as_of_date)
        for f in Path(data_dir).glob("*.parquet"):
            if f.stem in ('NIFTY50', 'NIFTYBANK', 'INDIAVIX'):
                continue
            try:
                df = pd.read_parquet(f, columns=['Close'])
                df.index = pd.to_datetime(df.index)
                c = df[df.index <= cutoff].sort_index()['Close'].dropna()
                closes[f.stem] = c.to_numpy()[-300:]
            except Exception:
                continue
        return rank_to_score(riskadj_scores(closes))

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   data_dir: str = "data/stocks") -> Tuple[float, float]:
        key = (as_of_date, data_dir)
        if cls._key != key:
            cls._scores = cls._build(as_of_date, data_dir)
            cls._key = key
        s = cls._scores.get(symbol.replace('.NS', '').upper())
        return (0.0, 0.0) if s is None else (float(s), 70.0)


# ==============================================================================
# QUALITY ALPHA  (profitability + low leverage, cross-sectional)
# ==============================================================================
class QualityAlpha:
    """
    Quality: profitable, lightly indebted companies.

    Score = average of two percentile ranks across the cached universe,
    mapped to [-1, 1]:
      - return on equity (return on assets for stocks Yahoo has no ROE for,
        ranked among themselves)
      - debt-to-equity, lower is better; skipped for financials, whose
        leverage is their business model

    Data: data/alpha_cache/quality.json from quality_data.py (monthly yfinance
    snapshots). Snapshot data is only valid going forward, so this alpha
    starts in the incubator and is judged by the shadow grade.
    Edge: quality tilts (e.g. NIFTY200 Quality 30) have a long record of
    smaller drawdowns, and screening momentum for quality avoids junk rallies.
    """

    MIN_UNIVERSE = 20
    FINANCIALS = {'Financial Services', 'Finance', 'Bank', 'Banks', 'NBFC',
                  'Insurance'}
    _scores: Optional[Dict[str, Tuple[float, float]]] = None

    @staticmethod
    def _pct_rank(values: Dict[str, float]) -> Dict[str, float]:
        if not values:
            return {}
        s = pd.Series(values, dtype=float)
        return ((s.rank() - 0.5) / len(s)).to_dict()

    @classmethod
    def build(cls, data: Dict[str, Dict]) -> Dict[str, Tuple[float, float]]:
        recs = {k: v for k, v in data.items() if not v.get('missing')}
        if len(recs) < cls.MIN_UNIVERSE:
            return {}
        roe = {k: v['roe'] for k, v in recs.items() if v.get('roe') is not None}
        roa = {k: v['roa'] for k, v in recs.items()
               if v.get('roe') is None and v.get('roa') is not None}
        de = {k: -v['de'] for k, v in recs.items()
              if v.get('de') is not None and v.get('sector') not in cls.FINANCIALS}
        prof = {**cls._pct_rank(roe), **cls._pct_rank(roa)}
        lev = cls._pct_rank(de)
        out = {}
        for sym in recs:
            parts = [r[sym] for r in (prof, lev) if sym in r]
            if parts:
                out[sym] = (float(np.clip((np.mean(parts) - 0.5) * 2, -1, 1)),
                            60.0 if len(parts) == 2 else 45.0)
        return out

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date) -> Tuple[float, float]:
        if cls._scores is None:
            try:
                from quality_data import load
                cls._scores = cls.build(load())
            except Exception:
                cls._scores = {}
        return cls._scores.get(symbol, (0.0, 0.0))


# ==============================================================================
# MASTER ALPHA AGGREGATOR  (UPDATED with 3 new signals)
# ==============================================================================
class IndiaAlphaAggregator:
    """
    Combines all India-specific alpha signals into a single composite score.

    Weights based on historical Sharpe ratios + India market characteristics:
      PEAD:           0.16  (strong academic backing in India)
      Momentum:       0.15  (persistent cross-sectional edge)
      FII/DII:        0.14  (market-wide flows predict direction)
      Mean Rev:       0.12  (high win rate on short horizon)
      Bulk Deal:      0.10  (informed accumulation)
      Delivery%:      0.08  (institutional vs speculative volume)
      Option Chain:   0.07  (PCR + max pain)
      Insider:        0.08  (high conviction but noisy)
      F&O Ban:        0.06  (mechanical, high precision, low frequency)
      Corp Event:     0.04  (catalyst-driven)

    Total = 1.00
    """

    WEIGHTS = {
        'pead':         0.16,
        # momentum 0.15 -> 0.17: the one price alpha our 10y walk-forward
        # PROVED (IC +0.026); absorbed the killed rel_strength weight.
        'momentum':     0.17,
        'fii_dii':      0.14,
        'mean_rev':     0.12,
        'bulk_deal':    0.10,
        'delivery_pct': 0.08,
        'option_chain': 0.07,
        'insider':      0.08,
        'fo_ban':       0.06,
        'corp_event':   0.04,
        # rel_strength KILLED 2026-06-11: 10y walk-forward rank IC NEGATIVE
        # (-0.011, walk_forward_report.md) — 63d leaders mildly revert over
        # the next 5 days. Class kept for reference; no longer scored.
        'hi52':         0.06,   # 52-week-high proximity (George & Hwang)
        'sector_mom':   0.06,   # sector rotation
        'pledge':       0.05,   # promoter pledge risk (new)
        'sast':         0.06,   # SAST stake changes (new)
        'shp_delta':    0.05,   # quarterly promoter stake delta (new)
        'quality':      0.06,   # ROE + low debt (incubating)
        'mom_riskadj':  0.17,   # NSE-style risk-adjusted momentum (incubating)
    }

    @classmethod
    def get_composite_signal(
        cls,
        symbol: str,
        as_of_date: date,
        data_dir: str = "data/stocks",
        universe_returns: Optional[Dict[str, float]] = None,
        verbose: bool = False
    ) -> Dict:
        """
        Compute all alpha signals and return composite.

        Returns dict with:
          score       : composite score [-1, 1]
          confidence  : composite confidence [0, 100]
          signal      : 'BUY' | 'SELL' | 'HOLD'
          components  : dict of individual alpha scores
        """
        components = {}
        weighted_score    = 0.0
        total_weight_used = 0.0

        alpha_funcs = {
            'pead':         lambda: PEADAlpha.get_signal(symbol, as_of_date),
            'momentum':     lambda: MomentumAlpha.get_signal(
                                symbol, as_of_date, data_dir, universe_returns),
            'fii_dii':      lambda: FIIDIIAlpha.get_signal(symbol, as_of_date),
            'mean_rev':     lambda: MeanRevAlpha.get_signal(
                                symbol, as_of_date, data_dir),
            'bulk_deal':    lambda: BulkDealAlpha.get_signal(symbol, as_of_date),
            'delivery_pct': lambda: DeliveryPercentAlpha.get_signal(
                                symbol, as_of_date),
            'option_chain': lambda: OptionChainAlpha.get_signal(
                                symbol, as_of_date),
            'insider':      lambda: InsiderAlpha.get_signal(symbol, as_of_date),
            'fo_ban':       lambda: FOBanAlpha.get_signal(symbol, as_of_date),
            'corp_event':   lambda: CorporateEventAlpha.get_signal(
                                symbol, as_of_date),
            'hi52':         lambda: Hi52Alpha.get_signal(
                                symbol, as_of_date, data_dir),
            'sector_mom':   lambda: SectorRotationAlpha.get_signal(
                                symbol, as_of_date, data_dir),
            'pledge':       lambda: PromoterPledgeAlpha.get_signal(
                                symbol, as_of_date),
            'sast':         lambda: SASTAlpha.get_signal(symbol, as_of_date),
            'shp_delta':    lambda: ShareholdingDeltaAlpha.get_signal(
                                symbol, as_of_date),
            'quality':      lambda: QualityAlpha.get_signal(symbol, as_of_date),
            'mom_riskadj':  lambda: RiskAdjMomentumAlpha.get_signal(
                                symbol, as_of_date, data_dir),
        }

        # Tier system (signal_decay_detector): live alphas = core +
        # promoted - disabled. Everything else runs in SHADOW mode — its
        # real score is computed and recorded (so the tracker keeps grading
        # it and it can earn promotion on evidence) but it contributes
        # ZERO weight to the composite.
        try:
            from signal_decay_detector import get_live_alphas
            live_alphas = get_live_alphas()
        except Exception:
            live_alphas = set(alpha_funcs.keys())   # fail-open: all live

        for name, func in alpha_funcs.items():
            try:
                score, conf = func()
                is_live = name in live_alphas
                components[name] = {
                    'score': round(float(score), 3),
                    'confidence': round(float(conf), 1),
                    'live': is_live,
                }
                if conf > 0 and is_live:
                    weighted_score    += score * cls.WEIGHTS[name]
                    total_weight_used += cls.WEIGHTS[name]
            except Exception as e:
                components[name] = {'score': 0.0, 'confidence': 0.0,
                                    'live': False, 'error': str(e)}

        # Normalise by weights used (graceful degradation)
        if total_weight_used > 0:
            composite_score = weighted_score / total_weight_used
        else:
            composite_score = 0.0

        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        # Composite confidence = average of non-zero LIVE confidences
        confs = [v['confidence'] for v in components.values()
                 if v['confidence'] > 0 and v.get('live', True)]
        composite_conf = float(np.mean(confs)) if confs else 0.0

        # Signal threshold
        if composite_score > 0.25:
            signal = 'BUY'
        elif composite_score < -0.25:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        result = {
            'symbol':     symbol,
            'date':       as_of_date.isoformat(),
            'score':      round(composite_score, 3),
            'confidence': round(composite_conf, 1),
            'signal':     signal,
            'components': components,
        }

        if verbose:
            print(f"\n  India Alpha: {symbol} ({as_of_date})")
            print(f"  Composite: {signal} | Score: {composite_score:+.3f}"
                  f" | Conf: {composite_conf:.0f}%")
            for name, v in components.items():
                bar = '#' * int(abs(v['score']) * 10)
                direction = '+' if v['score'] >= 0 else '-'
                err = f" [ERR: {v.get('error','')}]" if 'error' in v else ''
                print(f"  {name:12s}: {direction}{bar:<10} {v['score']:+.2f}"
                      f" ({v['confidence']:.0f}%){err}")

        return result


# ==============================================================================
# QUICK TEST
# ==============================================================================
if __name__ == '__main__':
    today = date.today()
    symbol = 'RELIANCE.NS'

    print("=" * 60)
    print(f"India Alpha Test: {symbol} as of {today}")
    print("=" * 60)

    result = IndiaAlphaAggregator.get_composite_signal(
        symbol=symbol,
        as_of_date=today,
        verbose=True
    )

    print(f"\nFinal: {result['signal']} | Score: {result['score']:+.3f} | Confidence: {result['confidence']:.0f}%")
