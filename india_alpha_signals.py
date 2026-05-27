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


def _safe_get(url: str, timeout: int = 10, headers: dict = None) -> Optional[requests.Response]:
    """Robust HTTP GET with retries."""
    _headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
        'Accept': 'application/json,text/html,*/*'
    }
    if headers:
        _headers.update(headers)
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=_headers, timeout=timeout)
            if resp.status_code == 200:
                return resp
        except Exception:
            pass
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

            records = []
            for entry in data:
                try:
                    records.append({
                        'date': pd.to_datetime(entry.get('date', entry.get('tradeDate', ''))),
                        'fii_buy': float(entry.get('fiiBuyValue', 0) or 0),
                        'fii_sell': float(entry.get('fiiSellValue', 0) or 0),
                        'dii_buy': float(entry.get('diiBuyValue', 0) or 0),
                        'dii_sell': float(entry.get('diiSellValue', 0) or 0),
                    })
                except Exception:
                    continue

            if not records:
                return None

            df = pd.DataFrame(records).set_index('date').sort_index()
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

    @classmethod
    def fetch_bulk_deals(cls, days_back: int = 30) -> Optional[pd.DataFrame]:
        """Fetch bulk deals from NSE."""
        today = date.today()
        cache_age_ok = (cls._cache_file.exists() and
                        (datetime.now() - datetime.fromtimestamp(cls._cache_file.stat().st_mtime)).seconds < 86400)

        if cache_age_ok:
            try:
                return pd.read_parquet(cls._cache_file)
            except Exception:
                pass

        # NSE bulk deal endpoint
        from_date = (today - timedelta(days=days_back)).strftime('%d-%m-%Y')
        to_date = today.strftime('%d-%m-%Y')
        url = (f"https://www.nseindia.com/api/bulk-deals?"
               f"from={from_date}&to={to_date}")

        resp = _safe_get(url, headers={'Referer': 'https://www.nseindia.com/'})
        if resp is None:
            return None

        try:
            data = resp.json()
            if not data or 'data' not in data:
                return None

            records = []
            for entry in data['data']:
                try:
                    records.append({
                        'date': pd.to_datetime(entry.get('BD_DT_DATE', '')),
                        'symbol': str(entry.get('BD_SYMBOL', '')).strip().upper(),
                        'client': str(entry.get('BD_CLIENT_NAME', '')),
                        'buy_sell': str(entry.get('BD_BUY_SELL', '')).upper(),
                        'quantity': float(entry.get('BD_QTY_TRD', 0) or 0),
                        'price': float(entry.get('BD_TP_WATP', 0) or 0),
                    })
                except Exception:
                    continue

            if not records:
                return None

            df = pd.DataFrame(records).set_index('date').sort_index()
            try:
                df.to_parquet(cls._cache_file)
            except Exception:
                pass
            return df

        except Exception as e:
            print(f"   [WARN] Bulk deals fetch error: {e}")
            return None

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
    SEBI insider trading disclosure signals.

    SEBI mandates insiders (promoters, directors, KMPs) to disclose
    all trades within 2 trading days. Research: promoter open-market
    purchases predict 3-12 month outperformance.

    Data: BSE corporate filings XML/JSON + SEBI EDIFAR (free, public)
    Edge: Information signal from people who know the company best.
    """

    _cache_file = _CACHE_DIR / "insider_trades.parquet"

    @classmethod
    def fetch_insider_trades(cls, days_back: int = 60) -> Optional[pd.DataFrame]:
        """Fetch insider trades from BSE bulk insider disclosures."""
        cache_age_ok = (cls._cache_file.exists() and
                        (datetime.now() - datetime.fromtimestamp(cls._cache_file.stat().st_mtime)).seconds < 86400 * 2)
        if cache_age_ok:
            try:
                return pd.read_parquet(cls._cache_file)
            except Exception:
                pass

        today = date.today()
        from_dt = (today - timedelta(days=days_back)).strftime('%Y%m%d')
        to_dt = today.strftime('%Y%m%d')

        # BSE insider trading data
        url = (f"https://api.bseindia.com/BseIndiaAPI/api/InsiderData/w?"
               f"dtfrom={from_dt}&dtto={to_dt}&type=i")

        resp = _safe_get(url, headers={'Referer': 'https://www.bseindia.com/'})
        if resp is None:
            return None

        try:
            data = resp.json()
            if not data:
                return None

            records = []
            for entry in data:
                try:
                    records.append({
                        'date': pd.to_datetime(entry.get('DT_TM', '')),
                        'symbol': str(entry.get('SCRIP_CD', '') or entry.get('SYMBOL', '')).strip().upper(),
                        'insider_name': str(entry.get('ACQNAME', '')),
                        'buy_sell': 'B' if float(entry.get('TOTACQSHRS', 0) or 0) > 0 else 'S',
                        'qty': float(entry.get('TOTACQSHRS', 0) or 0),
                        'value_cr': float(entry.get('ACQCOST', 0) or 0),
                    })
                except Exception:
                    continue

            if not records:
                return None

            df = pd.DataFrame(records).set_index('date').sort_index()
            try:
                df.to_parquet(cls._cache_file)
            except Exception:
                pass
            return df

        except Exception as e:
            return None  # Silently fail - BSE API unreliable

    @classmethod
    def get_signal(cls, symbol: str, as_of_date: date,
                   lookback_days: int = 30) -> Tuple[float, float]:
        """
        Returns (score, confidence).

        Positive score = promoter net buying in last 30 days.
        Negative score = promoter net selling (possible red flag).
        """
        df = cls.fetch_insider_trades(days_back=lookback_days + 10)
        if df is None:
            return 0.0, 0.0

        nse_sym = symbol.replace('.NS', '').upper()
        cutoff = as_of_date - timedelta(days=lookback_days)

        mask = (
            (df['symbol'].str.contains(nse_sym, na=False)) &
            (df.index.date >= cutoff) &
            (df.index.date <= as_of_date)
        )
        recent = df[mask]

        if len(recent) == 0:
            return 0.0, 0.0

        buy_val = recent[recent['buy_sell'] == 'B']['value_cr'].sum()
        sell_val = recent[recent['buy_sell'] == 'S']['value_cr'].sum()
        total_val = buy_val + sell_val

        if total_val == 0:
            return 0.0, 0.0

        score = (buy_val - sell_val) / total_val
        confidence = min(total_val / 10.0 * 10, 85.0)  # ₹10cr+ = max confidence

        return float(np.clip(score, -1.0, 1.0)), float(confidence)


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

        # Take most recent earnings event
        last = recent.iloc[-1]
        days_since = (as_of_date - recent.index[-1].date()).days

        # Look for surprise column (various names)
        surprise = None
        for col in ['surprise_pct', 'eps_surprise_pct', 'EPS_Surprise_Pct',
                    'Surprise_Pct', 'beat_pct']:
            if col in last.index and not pd.isna(last[col]):
                surprise = float(last[col])
                break

        if surprise is None:
            # Try to infer from beat_estimates boolean
            for col in ['beat_estimates', 'beat', 'Beat']:
                if col in last.index and not pd.isna(last[col]):
                    surprise = 8.0 if last[col] else -8.0
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

        url = "https://nseindia.com/api/fo-ban-list"
        resp = _safe_get(url, headers={'Referer': 'https://www.nseindia.com/'})

        if resp:
            try:
                data = resp.json()
                ban_symbols = set()
                if isinstance(data, dict) and 'data' in data:
                    for entry in data['data']:
                        sym = entry.get('symbol', '') or entry.get('SYMBOL', '')
                        if sym:
                            ban_symbols.add(str(sym).strip().upper())
                elif isinstance(data, list):
                    for entry in data:
                        sym = entry.get('symbol', '') if isinstance(entry, dict) else str(entry)
                        if sym:
                            ban_symbols.add(sym.strip().upper())

                # Save to cache
                try:
                    with open(cls._cache_file, 'w') as f:
                        import json
                        json.dump({'date': str(today), 'symbols': list(ban_symbols)}, f)
                except Exception:
                    pass

                cls._last_fetch = today
                cls._today_ban = ban_symbols
                return ban_symbols
            except Exception:
                pass

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
        'momentum':     0.15,
        'fii_dii':      0.14,
        'mean_rev':     0.12,
        'bulk_deal':    0.10,
        'delivery_pct': 0.08,
        'option_chain': 0.07,
        'insider':      0.08,
        'fo_ban':       0.06,
        'corp_event':   0.04,
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
        }

        # Check for disabled signals (from signal_decay_detector)
        disabled_signals: set = set()
        try:
            from signal_decay_detector import get_disabled_signals
            disabled_signals = set(get_disabled_signals())
        except Exception:
            pass

        for name, func in alpha_funcs.items():
            # Skip disabled signals
            if name in disabled_signals:
                components[name] = {'score': 0.0, 'confidence': 0.0,
                                    'disabled': True}
                continue
            try:
                score, conf = func()
                components[name] = {
                    'score': round(float(score), 3),
                    'confidence': round(float(conf), 1)
                }
                if conf > 0:
                    weighted_score    += score * cls.WEIGHTS[name]
                    total_weight_used += cls.WEIGHTS[name]
            except Exception as e:
                components[name] = {'score': 0.0, 'confidence': 0.0,
                                    'error': str(e)}

        # Normalise by weights used (graceful degradation)
        if total_weight_used > 0:
            composite_score = weighted_score / total_weight_used
        else:
            composite_score = 0.0

        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        # Composite confidence = average of non-zero confidences
        confs = [v['confidence'] for v in components.values()
                 if v['confidence'] > 0]
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
