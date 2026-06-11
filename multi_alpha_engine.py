"""
Multi-Alpha Meta-Learner Engine
================================
Combines ML model signals + India-specific alpha signals into a single
ranked, risk-adjusted recommendation with regime awareness.

Architecture:
  ┌──────────────────────────────────────────┐
  │           SIGNAL SOURCES                 │
  │  ML Ensemble  +  India Alpha Aggregator  │
  └──────────────┬───────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────┐
  │         META-LEARNER (XGBoost)           │
  │  Learns which signals work in each regime│
  └──────────────┬───────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────┐
  │        REGIME FILTER                     │
  │  Bull / Bear / Sideways / Crisis         │
  └──────────────┬───────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────┐
  │     CROSS-SECTIONAL RANKING              │
  │  Rank stocks by composite score          │
  │  Long top-5, avoid bottom-5              │
  └──────────────┬───────────────────────────┘
                 │
  ┌──────────────▼───────────────────────────┐
  │     PORTFOLIO RISK GATE                  │
  │  Correlation / Heat / Sector / Kelly     │
  └──────────────────────────────────────────┘
"""

import json
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

from india_alpha_signals import IndiaAlphaAggregator


# ==============================================================================
# IMPROVEMENT 1 — PER-STOCK PERFORMANCE DAMPENER
# ==============================================================================
class PerformanceDampener:
    """
    Dynamically adjusts a stock's composite score based on its recent 1-day
    win rate tracked in paper_trading/results/performance.csv.

    Formula:  multiplier = clip( win_rate / BASELINE, MIN_MULT, MAX_MULT )

    Examples:
      win_rate=20%  -> multiplier ~0.36  (strongly suppress)
      win_rate=45%  -> multiplier ~0.82  (mild suppress)
      win_rate=55%  -> multiplier ~1.00  (neutral — the baseline)
      win_rate=72%  -> multiplier ~1.15  (slight boost, capped)

    Requires MIN_TRADES before diverging from 1.0 so new stocks are unaffected.
    Self-correcting: as a stock's performance recovers the multiplier rises
    automatically without any manual intervention.
    """

    PERF_FILE  = Path("paper_trading/results/performance.csv")
    LOOKBACK   = 15     # Last N completed 1-day trades per stock
    MIN_TRADES = 5      # Need at least this many to apply dampening
    BASELINE   = 0.55   # Expected system win rate
    MIN_MULT   = 0.35   # Floor (worst persistent losers)
    MAX_MULT   = 1.15   # Cap  (avoid over-weighting lucky streaks)

    _cache: Optional[Dict[str, float]] = None
    _cache_date: Optional[date] = None

    @classmethod
    def get_multipliers(cls, as_of_date: Optional[date] = None) -> Dict[str, float]:
        """Returns {symbol: multiplier} for every stock with enough history."""
        today = as_of_date or date.today()
        if cls._cache_date == today and cls._cache is not None:
            return cls._cache

        if not cls.PERF_FILE.exists():
            return {}
        try:
            df = pd.read_csv(cls.PERF_FILE)
            if 'ret_1d' not in df.columns or 'symbol' not in df.columns:
                return {}
            df = df.dropna(subset=['ret_1d']).sort_values('signal_date')
            mults: Dict[str, float] = {}
            for sym, grp in df.groupby('symbol'):
                recent = grp.tail(cls.LOOKBACK)
                if len(recent) < cls.MIN_TRADES:
                    continue
                win_rate = float((recent['ret_1d'] > 0).mean())
                mult = float(np.clip(win_rate / cls.BASELINE,
                                     cls.MIN_MULT, cls.MAX_MULT))
                mults[str(sym)] = round(mult, 3)
            cls._cache = mults
            cls._cache_date = today
            return mults
        except Exception:
            return {}

    @classmethod
    def apply(cls, symbol: str, score: float,
              as_of_date: Optional[date] = None) -> Tuple[float, float]:
        """Returns (dampened_score, multiplier_used). 1.0 = no change."""
        mults = cls.get_multipliers(as_of_date)
        mult  = mults.get(symbol, 1.0)
        return float(np.clip(score * mult, -1.0, 1.0)), mult


# ==============================================================================
# IMPROVEMENT 3 — MARKET BREADTH DETECTOR
# ==============================================================================
class MarketBreadthDetector:
    """
    Computes advance/decline breadth across our universe from OHLCV parquets.

    A/D ratio < 0.35  -> VERY_WEAK  (broad selloff)  -> raise BUY bar sharply
    A/D ratio < 0.45  -> WEAK       (soft day)        -> raise BUY bar
    A/D ratio < 0.55  -> SOFT       (mixed)           -> slight raise
    A/D ratio 0.55-65 -> NORMAL                       -> no change
    A/D ratio > 0.65  -> STRONG     (broad rally)     -> slight ease

    Also tracks % of stocks above their 5-day moving average.
    Result is cached per date so the parquet scan runs only once per day.
    """

    _cache: Dict[str, Dict] = {}

    @classmethod
    def compute(cls, as_of_date: date, symbols: List[str],
                data_dir: str = "data/stocks") -> Dict:
        key = str(as_of_date)
        if key in cls._cache:
            return cls._cache[key]

        advances = declines = above_ma5 = total = 0

        for sym in symbols:
            nse = sym.replace('.NS', '').upper()
            fpath = Path(data_dir) / f"{nse}.parquet"
            if not fpath.exists():
                continue
            try:
                df = pd.read_parquet(fpath)
                df.index = pd.to_datetime(df.index)
                df = (df[df.index.date <= as_of_date]
                        .sort_index()
                        .dropna(subset=['Close']))
                if len(df) < 6:
                    continue
                today_c = float(df['Close'].iloc[-1])
                prev_c  = float(df['Close'].iloc[-2])
                ma5     = float(df['Close'].tail(5).mean())
                total  += 1
                if today_c > prev_c:
                    advances += 1
                elif today_c < prev_c:
                    declines += 1
                if today_c > ma5:
                    above_ma5 += 1
            except Exception:
                continue

        if total == 0:
            out = {'ad_ratio': 0.5, 'pct_above_ma5': 0.5,
                   'score_scale': 1.0, 'label': 'UNKNOWN',
                   'advances': 0, 'declines': 0, 'total': 0}
            cls._cache[key] = out
            return out

        ad   = advances / total
        pma5 = above_ma5 / total

        if ad < 0.35:
            scale, label = 0.55, 'VERY_WEAK'
        elif ad < 0.45:
            scale, label = 0.72, 'WEAK'
        elif ad < 0.55:
            scale, label = 0.88, 'SOFT'
        elif ad < 0.65:
            scale, label = 1.00, 'NORMAL'
        else:
            scale, label = 1.08, 'STRONG'

        out = {
            'ad_ratio':      round(ad, 3),
            'pct_above_ma5': round(pma5, 3),
            'score_scale':   scale,
            'label':         label,
            'advances':      advances,
            'declines':      declines,
            'total':         total,
        }
        cls._cache[key] = out
        return out


# ==============================================================================
# IMPROVEMENT 4 — PER-STOCK SIGNAL WEIGHT LEARNER
# ==============================================================================
class StockWeightLearner:
    """
    Maintains per-stock multipliers for each of the 10 alpha signals.

    When an outcome is known (paper_trade_tracker fetches ret_1d), the engine
    replays: for each signal that had an opinion, did it agree with the result?
    - Agreed  -> EMA toward 1.5 (boost that signal for this stock)
    - Disagreed -> EMA toward 0.5 (dampen that signal for this stock)

    EMA formula:  new = EMA_ALPHA * old + (1-EMA_ALPHA) * target

    Multipliers start at 1.0 and blend in gradually:
      < MIN_TRADES        -> 0% learned (pure global weights)
      MIN_TRADES..30      -> linearly up to 70% learned
      >= 30 trades        -> 70% learned, 30% global (never fully override)

    Storage:  paper_trading/stock_weights.json
    Called:   paper_trade_tracker.py after each outcome batch
    """

    WEIGHTS_FILE      = Path("paper_trading/stock_weights.json")
    EMA_ALPHA         = 0.85   # Slow learner — needs consistent evidence
    MIN_TRADES        = 5
    FULL_BLEND_TRADES = 30
    MAX_MULT          = 2.0
    MIN_MULT          = 0.15

    SIGNALS = ['momentum', 'fii_dii', 'pead', 'mean_rev', 'bulk_deal',
               'delivery_pct', 'option_chain', 'insider', 'fo_ban', 'corp_event',
               'rel_strength', 'sector_mom', 'pledge', 'sast', 'shp_delta']

    _data: Optional[Dict] = None

    @classmethod
    def _load(cls) -> Dict:
        if cls._data is not None:
            return cls._data
        cls.WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        if cls.WEIGHTS_FILE.exists():
            try:
                with open(cls.WEIGHTS_FILE) as f:
                    cls._data = json.load(f)
                return cls._data
            except Exception:
                pass
        cls._data = {}
        return cls._data

    @classmethod
    def _save(cls):
        try:
            cls.WEIGHTS_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(cls.WEIGHTS_FILE, 'w') as f:
                json.dump(cls._data, f, indent=2)
        except Exception:
            pass

    @classmethod
    def get_multipliers(cls, symbol: str) -> Dict[str, float]:
        """
        Returns {signal_name: multiplier} for this stock.
        Falls back gracefully to 1.0 for all signals if no data yet.
        """
        data     = cls._load()
        sd       = data.get(symbol, {})
        learned  = sd.get('multipliers', {})
        # [Overfitting guard] Trust each signal based on ITS OWN evidence count,
        # not the stock's total trades. A signal that has only fired a few times
        # stays near-neutral even if the stock has many trades overall — this
        # stops the learner from chasing noise on thin per-signal samples.
        counts   = sd.get('sig_counts', {})

        out = {}
        for s in cls.SIGNALS:
            c = counts.get(s, 0)
            if c < cls.MIN_TRADES:
                out[s] = 1.0
                continue
            blend = min(
                0.70 * (c - cls.MIN_TRADES) /
                max(cls.FULL_BLEND_TRADES - cls.MIN_TRADES, 1),
                0.70
            )
            out[s] = round((1 - blend) * 1.0 + blend * learned.get(s, 1.0), 3)
        return out

    @classmethod
    def update_from_outcome(cls, symbol: str,
                            alpha_scores: Dict,
                            ret_1d: float):
        """
        Update per-stock signal multipliers from one known 1-day outcome.
        alpha_scores: {'signal_name': {'score': x, 'confidence': y}, ...}
        """
        if not alpha_scores or ret_1d is None or np.isnan(float(ret_1d)):
            return
        data = cls._load()
        if symbol not in data:
            data[symbol] = {'n_trades': 0, 'multipliers': {}}
        sd = data[symbol]
        sd['n_trades'] = sd.get('n_trades', 0) + 1
        mults = sd.setdefault('multipliers', {})
        counts = sd.setdefault('sig_counts', {})   # per-signal evidence count
        pos_outcome = float(ret_1d) > 0

        for sig in cls.SIGNALS:
            comp  = alpha_scores.get(sig, {})
            score = comp.get('score', 0.0) if isinstance(comp, dict) else 0.0
            conf  = comp.get('confidence', 0.0) if isinstance(comp, dict) else 0.0
            if conf <= 0 or abs(score) < 0.05:
                continue
            counts[sig] = counts.get(sig, 0) + 1   # this signal had a real view
            agreed = (score > 0) == pos_outcome
            target = 1.5 if agreed else 0.5
            old    = mults.get(sig, 1.0)
            mults[sig] = round(float(np.clip(
                cls.EMA_ALPHA * old + (1 - cls.EMA_ALPHA) * target,
                cls.MIN_MULT, cls.MAX_MULT
            )), 3)
        cls._save()

    @classmethod
    def batch_update(cls, perf_df: pd.DataFrame,
                     alpha_dir: str = "paper_trading/alpha_scores"):
        """
        Batch update from performance DataFrame.
        Reads per-trade alpha score files saved by the engine.
        """
        adir = Path(alpha_dir)
        if not adir.exists():
            return 0
        updated = 0
        for _, row in perf_df.iterrows():
            ret_1d = row.get('ret_1d')
            if pd.isna(ret_1d):
                continue
            sym  = str(row.get('symbol', ''))
            sdate = str(row.get('signal_date', ''))
            safe = sym.replace('.NS', '').replace('/', '_')
            sf   = adir / f"{safe}_{sdate}.json"
            if not sf.exists():
                continue
            try:
                with open(sf) as f:
                    alpha_scores = json.load(f)
                cls.update_from_outcome(sym, alpha_scores, float(ret_1d))
                updated += 1
            except Exception:
                continue
        if updated:
            print(f"   [WeightLearner] Updated {updated} stock-signal weights")
        return updated


# ==============================================================================
# REGIME DETECTOR
# ==============================================================================
class RegimeDetector:
    """
    Detects current market regime using NIFTY 50 + India VIX.

    Regimes:
      BULL      — Trending up, low volatility → momentum works
      BEAR      — Trending down, rising VIX → defensive, reduce size
      SIDEWAYS  — Low trend, moderate vol → mean reversion works
      CRISIS    — VIX spike (>25), rapid falls → cash / reduce all

    Uses: Nifty 50 20/50 MA ratio + India VIX level + VIX 5d change
    """

    @staticmethod
    def detect(as_of_date: date, data_dir: str = "data/stocks") -> Dict:
        """
        Returns regime dict:
          regime    : str  ('BULL' | 'BEAR' | 'SIDEWAYS' | 'CRISIS')
          confidence: float (0-100)
          vix       : float
          nifty_trend: float  (20-day SMA / 50-day SMA - 1)
        """
        try:
            # Load NIFTY 50 data
            nifty_path = Path(data_dir) / "NIFTY50.parquet"
            if not nifty_path.exists():
                nifty_path = Path(data_dir) / "NIFTY50-INDEX.parquet"

            vix_path = Path(data_dir) / "INDIAVIX.parquet"

            nifty_df = None
            vix_value = 15.0  # Default moderate

            if nifty_path.exists():
                nifty_df = pd.read_parquet(nifty_path)
                nifty_df.index = pd.to_datetime(nifty_df.index)
                nifty_df = nifty_df[nifty_df.index.date <= as_of_date].sort_index()

            if vix_path.exists():
                vix_df = pd.read_parquet(vix_path)
                vix_df.index = pd.to_datetime(vix_df.index)
                vix_df = vix_df[vix_df.index.date <= as_of_date].sort_index()
                if len(vix_df) > 0:
                    vix_value = float(vix_df['Close'].iloc[-1])

            # Default regime if data unavailable
            if nifty_df is None or len(nifty_df) < 50:
                return {
                    'regime': 'SIDEWAYS', 'confidence': 30.0,
                    'vix': vix_value, 'nifty_trend': 0.0
                }

            close = nifty_df['Close']
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            nifty_trend = (sma20 / sma50 - 1) if sma50 > 0 else 0.0

            # 5-day VIX change
            if vix_path.exists() and len(vix_df) > 5:
                vix_5d_change = (float(vix_df['Close'].iloc[-1]) /
                                 float(vix_df['Close'].iloc[-6]) - 1)
            else:
                vix_5d_change = 0.0

            # Short-term Nifty 3-day change (for transition detection)
            nifty_3d_chg = 0.0
            if len(nifty_df) >= 4:
                c_now  = float(close.iloc[-1])
                c_3ago = float(close.iloc[-4])
                nifty_3d_chg = (c_now / c_3ago - 1) if c_3ago > 0 else 0.0

            # Regime classification
            if vix_value > 25 or vix_5d_change > 0.25:
                regime     = 'CRISIS'
                confidence = min(vix_value * 3, 95.0)
            elif nifty_trend > 0.015 and vix_value < 18:
                regime     = 'BULL'
                confidence = min(abs(nifty_trend) * 1000, 85.0)
            elif nifty_trend < -0.015:
                regime     = 'BEAR'
                confidence = min(abs(nifty_trend) * 1000, 80.0)
            else:
                regime     = 'SIDEWAYS'
                confidence = 60.0

            # ── Improvement 5: Regime transition probability ─────────────────
            # Detects whether the regime is stable or about to flip.
            # DETERIORATING/WEAKENING → raise BUY bar in rank_universe().
            regime_momentum = 'STABLE'
            transition_risk = 0.10

            if regime == 'SIDEWAYS':
                if nifty_3d_chg < -0.008 and vix_5d_change > 0.03:
                    regime_momentum = 'DETERIORATING'
                    transition_risk = 0.45
                elif nifty_3d_chg < -0.005 or vix_5d_change > 0.08:
                    regime_momentum = 'WEAKENING'
                    transition_risk = 0.30
                elif nifty_3d_chg > 0.008 and vix_5d_change < -0.03:
                    regime_momentum = 'IMPROVING'
                    transition_risk = 0.08
            elif regime == 'BULL':
                if nifty_3d_chg < -0.015 or vix_5d_change > 0.20:
                    regime_momentum = 'DETERIORATING'
                    transition_risk = 0.40
                elif nifty_3d_chg < -0.007:
                    regime_momentum = 'WEAKENING'
                    transition_risk = 0.22
            elif regime == 'BEAR':
                if nifty_3d_chg > 0.010 and vix_5d_change < 0.0:
                    regime_momentum = 'RECOVERING'
                    transition_risk = 0.28
            elif regime == 'CRISIS':
                if nifty_3d_chg > 0.015 and vix_5d_change < -0.10:
                    regime_momentum = 'RECOVERING'
                    transition_risk = 0.30

            return {
                'regime':          regime,
                'confidence':      round(confidence, 1),
                'vix':             round(vix_value, 2),
                'nifty_trend':     round(nifty_trend * 100, 2),
                'nifty_3d_change': round(nifty_3d_chg * 100, 2),
                'regime_momentum': regime_momentum,
                'transition_risk': round(transition_risk, 2),
            }

        except Exception as e:
            return {
                'regime': 'SIDEWAYS', 'confidence': 20.0,
                'vix': 15.0, 'nifty_trend': 0.0,
                'nifty_3d_change': 0.0,
                'regime_momentum': 'STABLE', 'transition_risk': 0.10,
            }


# ==============================================================================
# MULTI-ALPHA ENGINE
# ==============================================================================
class MultiAlphaEngine:
    """
    Master engine: fetches all alpha signals, applies regime filter,
    cross-sectional ranking, and risk gates.
    """

    # Regime-specific alpha weights — what works in each regime
    # New signals added: delivery_pct, option_chain, corp_event
    REGIME_WEIGHTS = {
        'BULL': {
            'ml_score':     0.27,  # ML trend-following dominant
            'momentum':     0.22,  # 12-1 momentum shines in bull markets
            'fii_dii':      0.13,  # Flow momentum confirms trend
            'pead':         0.13,  # Earnings beat follow-through
            'mean_rev':     0.04,  # Less useful in strong trends
            'bulk_deal':    0.05,
            'insider':      0.03,
            'delivery_pct': 0.05,  # Institutional accumulation confirmation
            'option_chain': 0.05,  # PCR + max pain
            'corp_event':   0.03,  # Catalyst signals
        },
        'BEAR': {
            'ml_score':     0.22,
            'momentum':     0.04,  # Momentum destroys capital in bear markets
            'fii_dii':      0.17,  # FII selling is the bear driver
            'pead':         0.08,
            'mean_rev':     0.22,  # Bear market bounces
            'bulk_deal':    0.08,
            'insider':      0.04,
            'delivery_pct': 0.06,  # Distribution pattern shows real selling
            'option_chain': 0.06,  # PCR extremes predict reversals
            'corp_event':   0.03,
        },
        'SIDEWAYS': {
            'ml_score':     0.22,
            'momentum':     0.08,
            'fii_dii':      0.10,
            'pead':         0.15,
            'mean_rev':     0.17,  # Mean reversion dominant
            'bulk_deal':    0.09,
            'insider':      0.04,
            'delivery_pct': 0.06,
            'option_chain': 0.06,
            'corp_event':   0.03,
        },
        'CRISIS': {
            'ml_score':     0.13,  # ML unreliable in extreme events
            'momentum':     0.04,
            'fii_dii':      0.26,  # FII flows are the dominant factor
            'pead':         0.04,
            'mean_rev':     0.26,  # Deep oversold bounces
            'bulk_deal':    0.08,
            'insider':      0.04,
            'delivery_pct': 0.07,  # Panic selling vs institutional buying
            'option_chain': 0.05,  # PCR extremes signal reversal
            'corp_event':   0.03,
        },
    }

    def __init__(
        self,
        data_dir: str = "data/stocks",
        model_cache_dir: str = "model_cache",
        min_composite_score: float = 0.30,  # Min abs score to act
        min_confidence: float = 55.0,       # Min confidence to act
    ):
        self.data_dir = data_dir
        self.model_cache_dir = Path(model_cache_dir)
        self.min_composite_score = min_composite_score
        self.min_confidence = min_confidence

        # Load meta-learner if available
        self._meta_model = self._load_meta_model()

    def _load_meta_model(self):
        """Load pre-trained meta-learner from cache."""
        meta_path = self.model_cache_dir / "meta_learner.joblib"
        if meta_path.exists():
            try:
                return joblib.load(meta_path)
            except Exception:
                pass
        return None

    def _save_meta_model(self, model):
        """Save meta-learner to cache."""
        self.model_cache_dir.mkdir(exist_ok=True)
        meta_path = self.model_cache_dir / "meta_learner.joblib"
        try:
            joblib.dump(model, meta_path)
        except Exception:
            pass

    def _get_ml_score(self, symbol: str, as_of_date: date) -> Tuple[float, float]:
        """
        Fetch ML model prediction from cached model outputs or run fresh.
        Returns (score [-1,1], confidence [0,100]).
        """
        nse_sym = symbol.replace('.NS', '').upper()
        ml_cache_path = self.model_cache_dir / f"{nse_sym}_latest_pred.joblib"

        if ml_cache_path.exists():
            try:
                cached = joblib.load(ml_cache_path)
                if cached.get('date') == str(as_of_date):
                    return cached.get('score', 0.0), cached.get('confidence', 0.0)
            except Exception:
                pass

        # No cached prediction — return neutral (will be filled by daily_runner)
        return 0.0, 0.0

    def score_stock(
        self,
        symbol: str,
        as_of_date: date,
        regime: str = 'SIDEWAYS',
        universe_returns: Optional[Dict[str, float]] = None,
        ml_score: float = 0.0,
        ml_confidence: float = 0.0,
    ) -> Dict:
        """
        Score a single stock using all alphas and the regime-specific weights.

        Returns dict with composite score, signal, confidence, breakdown.
        """
        weights = self.REGIME_WEIGHTS.get(regime, self.REGIME_WEIGHTS['SIDEWAYS'])

        # ── India Alpha Signals ─────────────────────────────────────────────
        alpha_result = IndiaAlphaAggregator.get_composite_signal(
            symbol=symbol,
            as_of_date=as_of_date,
            data_dir=self.data_dir,
            universe_returns=universe_returns,
        )
        components = alpha_result.get('components', {})

        # ── [Improvement 4] Per-stock signal weight multipliers ──────────────
        # Starts at 1.0 for all signals; blends in learned values once
        # >= MIN_TRADES outcomes are recorded for this stock.
        stock_mults = StockWeightLearner.get_multipliers(symbol)

        # ── Combine with regime weights ──────────────────────────────────────
        weighted_score = 0.0
        total_weight   = 0.0
        signals_with_view: List[Tuple[str, float]] = []   # (name, score)

        # ML score contribution (regime-controlled, no per-stock learning)
        if ml_confidence > 0:
            ml_normalised   = float(np.clip(ml_score, -1.0, 1.0))
            weighted_score += ml_normalised * weights['ml_score']
            total_weight   += weights['ml_score']

        # India alpha contributions
        alpha_name_map = {
            'momentum':     'momentum',
            'fii_dii':      'fii_dii',
            'pead':         'pead',
            'mean_rev':     'mean_rev',
            'bulk_deal':    'bulk_deal',
            'insider':      'insider',
            'fo_ban':       'fo_ban',
            'delivery_pct': 'delivery_pct',
            'option_chain': 'option_chain',
            'corp_event':   'corp_event',
            'rel_strength': 'rel_strength',
            'sector_mom':   'sector_mom',
            'pledge':       'pledge',
            'sast':         'sast',
            'shp_delta':    'shp_delta',
        }
        for alpha_key, weight_key in alpha_name_map.items():
            comp  = components.get(alpha_key, {})
            score = comp.get('score', 0.0)
            conf  = comp.get('confidence', 0.0)
            if conf > 0:
                # [Improvement 4] scale regime weight by per-stock multiplier.
                # Alphas absent from a regime map (e.g. fo_ban) get a small
                # default so they still contribute; normalisation handles sums.
                eff_w           = weights.get(weight_key, 0.04) * stock_mults.get(alpha_key, 1.0)
                weighted_score += score * eff_w
                total_weight   += eff_w
            if conf > 0 and abs(score) > 0.05:
                signals_with_view.append((alpha_key, score))

        # Normalise
        composite_score = weighted_score / total_weight if total_weight > 0 else 0.0
        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        # ── [Improvement 2] Cross-signal consensus gate ──────────────────────
        # Penalise calls driven by a single dominant signal while most others
        # disagree. Requires >= 3 signals with an opinion to apply.
        consensus_ratio = 1.0
        if len(signals_with_view) >= 3:
            agreeing = sum(
                1 for _, s in signals_with_view
                if (s * composite_score) > 0
            )
            consensus_ratio = agreeing / len(signals_with_view)
            if consensus_ratio < 0.25:
                composite_score *= 0.50   # strong penalty — lone-wolf signal
            elif consensus_ratio < 0.40:
                composite_score *= 0.75   # moderate penalty
            # consensus >= 0.40 → no penalty; score stands as-is

        # ── [Improvement 1] Per-stock performance dampening ─────────────────
        composite_score, dampen_mult = PerformanceDampener.apply(
            symbol, composite_score, as_of_date
        )

        # Composite confidence
        confs = [c.get('confidence', 0) for c in components.values()
                 if c.get('confidence', 0) > 0]
        if ml_confidence > 0:
            confs.append(ml_confidence)
        composite_conf = float(np.mean(confs)) if confs else 0.0

        # Signal determination
        if composite_score > self.min_composite_score and composite_conf >= self.min_confidence:
            signal = 'BUY'
        elif composite_score < -self.min_composite_score and composite_conf >= self.min_confidence:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # CRISIS override: never BUY unless very strong conviction
        if regime == 'CRISIS' and composite_score < 0.6:
            signal = 'HOLD'

        return {
            'symbol':               symbol,
            'date':                 as_of_date.isoformat(),
            'signal':               signal,
            'composite_score':      round(composite_score, 3),
            'composite_confidence': round(composite_conf, 1),
            'regime':               regime,
            'ml_score':             round(ml_score, 3),
            'ml_confidence':        round(ml_confidence, 1),
            'alpha_components':     components,
            'weight_used':          round(total_weight, 3),
            'consensus_ratio':      round(consensus_ratio, 2),
            'dampen_multiplier':    round(dampen_mult, 3),
        }

    def rank_universe(
        self,
        symbols: List[str],
        as_of_date: date,
        ml_scores: Optional[Dict[str, Tuple[float, float]]] = None,
        top_n_long: int = 5,
        top_n_avoid: int = 5,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Score all stocks, apply regime filter, cross-sectionally rank.

        Returns DataFrame sorted by composite_score descending.
        Marks top_n_long as BUY candidates, bottom top_n_avoid as AVOID.
        """
        # ── Detect regime ────────────────────────────────────────────────────
        regime_info     = RegimeDetector.detect(as_of_date, self.data_dir)
        regime          = regime_info['regime']
        regime_conf     = regime_info['confidence']
        regime_momentum = regime_info.get('regime_momentum', 'STABLE')
        transition_risk = regime_info.get('transition_risk', 0.10)
        nifty_3d        = regime_info.get('nifty_3d_change', 0.0)

        # ── [Improvement 3] Market breadth ───────────────────────────────────
        breadth = MarketBreadthDetector.compute(as_of_date, symbols, self.data_dir)

        # ── [Improvement 5] Dynamic threshold from breadth + transition risk ─
        # breadth.score_scale:  <1 = raise bar, >1 = ease bar
        # transition_mult:      <1 = raise bar when regime deteriorating
        if regime_momentum == 'DETERIORATING':
            transition_mult = 0.68
        elif regime_momentum == 'WEAKENING':
            transition_mult = 0.84
        elif regime_momentum == 'IMPROVING':
            transition_mult = 1.05
        else:
            transition_mult = 1.00

        # Combined: composite_score must beat min_score/effective_scale to qualify
        effective_scale = breadth['score_scale'] * transition_mult

        if verbose:
            print(f"\n{'='*60}")
            print(f"[REGIME] {regime} ({regime_conf:.0f}% conf) | "
                  f"VIX {regime_info['vix']:.1f} | "
                  f"Nifty 3d {nifty_3d:+.1f}%")
            if regime_momentum != 'STABLE':
                print(f"   [!] Regime {regime_momentum} "
                      f"(transition risk {transition_risk:.0%})")
            bl = breadth['label']
            print(f"[BREADTH] {bl}: {breadth['ad_ratio']:.0%} advancing, "
                  f"{breadth['pct_above_ma5']:.0%} above MA5 "
                  f"({breadth['advances']}up/{breadth['declines']}dn "
                  f"of {breadth['total']})")
            if effective_scale != 1.0:
                print(f"[ADAPT]  Score bar scaled x{effective_scale:.2f} "
                      f"(breadth={breadth['score_scale']:.2f} x "
                      f"regime={transition_mult:.2f})")
            print(f"{'='*60}")

        # ── Compute universe 12-1 month returns for cross-sectional momentum ─
        universe_returns: Dict[str, float] = {}
        for sym in symbols:
            nse_sym = sym.replace('.NS', '').upper()
            fpath = Path(self.data_dir) / f"{nse_sym}.parquet"
            if fpath.exists():
                try:
                    df = pd.read_parquet(fpath)
                    df.index = pd.to_datetime(df.index)
                    df = df[df.index.date <= as_of_date].sort_index()
                    if len(df) >= 252:
                        ret = df['Close'].iloc[-22] / df['Close'].iloc[-252] - 1
                        universe_returns[nse_sym] = float(ret)
                except Exception:
                    pass

        # ── Score each stock ─────────────────────────────────────────────────
        rows = []
        for i, symbol in enumerate(symbols):
            if verbose and i % 10 == 0:
                print(f"   Scoring {i+1}/{len(symbols)}: {symbol}...")

            ml_s, ml_c = (ml_scores.get(symbol, (0.0, 0.0))
                          if ml_scores else (0.0, 0.0))

            result = self.score_stock(
                symbol=symbol,
                as_of_date=as_of_date,
                regime=regime,
                universe_returns=universe_returns,
                ml_score=ml_s,
                ml_confidence=ml_c,
            )
            rows.append(result)

        df = pd.DataFrame(rows)
        df = df.sort_values('composite_score', ascending=False).reset_index(drop=True)

        # ── [Improvement 5+3] Apply effective_scale to signal thresholds ─────
        # Instead of changing min_composite_score globally (which affects
        # score_stock already called), we re-evaluate signals here using the
        # scaled threshold.  effective_scale < 1 → harder to be a BUY.
        adj_min = self.min_composite_score / max(effective_scale, 0.1)
        for idx in df.index:
            raw_score = df.at[idx, 'composite_score']
            raw_conf  = df.at[idx, 'composite_confidence']
            cur_sig   = df.at[idx, 'signal']
            if cur_sig == 'BUY' and raw_score < adj_min:
                df.at[idx, 'signal'] = 'HOLD'   # score not strong enough after scaling
            elif cur_sig == 'HOLD' and raw_score > adj_min and raw_conf >= self.min_confidence:
                df.at[idx, 'signal'] = 'BUY'    # score now qualifies after easing
            # SELL signals: scale doesn't apply to short side (no breadth logic there)

        # ── Cross-sectional ranking labels ───────────────────────────────────
        df['rank'] = range(1, len(df) + 1)
        df['cs_label'] = 'NEUTRAL'

        # Top N → BUY candidates
        top_mask = df['rank'] <= top_n_long
        df.loc[top_mask & (df['composite_score'] > 0.15), 'cs_label'] = 'TOP_BUY'

        # Bottom N → AVOID
        bottom_mask = df['rank'] > (len(df) - top_n_avoid)
        df.loc[bottom_mask & (df['composite_score'] < -0.15), 'cs_label'] = 'AVOID'

        # Upgrade HOLD → BUY only if score also passes scaled threshold
        df.loc[
            (df['cs_label'] == 'TOP_BUY') &
            (df['signal'] == 'HOLD') &
            (df['composite_score'] >= adj_min),
            'signal'
        ] = 'BUY'

        # ── Save alpha scores for weight-learning feedback loop ───────────────
        # paper_trade_tracker will read these files to update StockWeightLearner
        alpha_dir = Path("paper_trading/alpha_scores")
        alpha_dir.mkdir(parents=True, exist_ok=True)
        for _, row in df.iterrows():
            try:
                sym   = str(row['symbol']).replace('.NS','').replace('/','_')
                fname = alpha_dir / f"{sym}_{as_of_date}.json"
                comps = row.get('alpha_components', {})
                if isinstance(comps, dict) and comps:
                    with open(fname, 'w') as fj:
                        json.dump(comps, fj)
            except Exception:
                pass

        # Add regime + breadth metadata
        df['regime']           = regime
        df['regime_confidence']= regime_conf
        df['regime_momentum']  = regime_momentum
        df['transition_risk']  = transition_risk
        df['vix']              = regime_info['vix']
        df['breadth_label']    = breadth['label']
        df['breadth_ad_ratio'] = breadth['ad_ratio']

        if verbose:
            print(f"\n📊 Top 5 BUY Candidates:")
            top5 = df.head(top_n_long)[['symbol', 'signal', 'composite_score', 'composite_confidence']]
            for _, row in top5.iterrows():
                print(f"   {row['symbol']:20s} {row['signal']:5s} "
                      f"Score: {row['composite_score']:+.2f}  Conf: {row['composite_confidence']:.0f}%")

            sell_df = df[df['signal'] == 'SELL']
            if len(sell_df) > 0:
                print(f"\n📉 Top SELL Signals:")
                for _, row in sell_df.head(3).iterrows():
                    print(f"   {row['symbol']:20s} {row['signal']:5s} "
                          f"Score: {row['composite_score']:+.2f}  Conf: {row['composite_confidence']:.0f}%")

        return df

    def train_meta_learner(
        self,
        historical_scores: pd.DataFrame,
        historical_returns: pd.Series,
        min_samples: int = 200
    ):
        """
        Train a meta-learner (XGBoost) to predict which alpha combinations
        lead to the best 5-day forward returns.

        historical_scores: DataFrame with alpha component scores per row
        historical_returns: 5-day forward returns (aligned with scores)
        """
        if not XGB_AVAILABLE:
            print("   ⚠️ XGBoost not available — meta-learner skipped")
            return

        if len(historical_scores) < min_samples:
            print(f"   ⚠️ Not enough data for meta-learner ({len(historical_scores)} < {min_samples})")
            return

        # Feature matrix
        feature_cols = [c for c in historical_scores.columns
                       if c not in {'symbol', 'date', 'signal', 'regime', 'cs_label'}
                       and historical_scores[c].dtype in (np.float64, np.float32)]

        X = historical_scores[feature_cols].fillna(0)
        y = (historical_returns > 0.02).astype(int)  # BUY if >2% in 5 days

        # Time-series split
        split = int(len(X) * 0.8)
        X_tr, X_val = X.iloc[:split], X.iloc[split:]
        y_tr, y_val = y.iloc[:split], y.iloc[split:]

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='logloss',
            random_state=42, verbosity=0
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        val_acc = (model.predict(X_val) == y_val).mean()
        print(f"   ✅ Meta-learner trained: val accuracy = {val_acc:.1%}")

        self._meta_model = model
        self._save_meta_model(model)


# ==============================================================================
# QUICK TEST
# ==============================================================================
if __name__ == '__main__':
    engine = MultiAlphaEngine()
    today = date.today()

    symbols = [
        'RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'AXISBANK.NS', 'LT.NS'
    ]

    print("Multi-Alpha Engine Test")
    df = engine.rank_universe(symbols, today, verbose=True)
    print(f"\n✅ Ranked {len(df)} stocks")
    print(df[['symbol', 'signal', 'composite_score', 'composite_confidence',
               'cs_label', 'regime']].to_string(index=False))
