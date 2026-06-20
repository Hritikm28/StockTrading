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
# NOTE: the per-stock PerformanceDampener was REMOVED 2026-06-12. It scaled
# scores by each stock's recent win rate, which is procyclical: a drawdown
# dampened scores, which produced worse entries, which extended the drawdown.
# System-level protection now lives in circuit_breaker.py (portfolio-level,
# with statistical evidence) and the NIFTY 200-DMA trend gate below.
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
      MIN_TRADES..60      -> linearly up to 50% learned
      >= 60 trades        -> 50% learned, 50% global (never fully override)

    [2026-06-12] MIN_TRADES raised 5 -> 30, blend capped at 50%.
    [2026-06-19] MIN_TRADES 30 -> 12 with a TIGHTER target band and a smaller
    blend cap. 30 outcomes *per signal per stock* is unreachable in a ~60-stock
    universe emitting ~8 signals/day — the most-traded name had 12 TOTAL trades —
    so the learner was permanently frozen at neutral (it "never learned from its
    mistakes"). Instead we let it start nudging at 12 outcomes but make each nudge
    gentle: the per-outcome target is 1.25/0.75 (was 1.5/0.5) and the global blend
    is capped at 35% (was 50%), so even a stock that always agrees/disagrees can
    only move its effective weight to roughly [0.90, 1.10]. Gentle, bounded
    adaptation — not noise-chasing swings.

    Storage:  paper_trading/stock_weights.json
    Called:   paper_trade_tracker.py after each outcome batch
    """

    WEIGHTS_FILE      = Path("paper_trading/stock_weights.json")
    EMA_ALPHA         = 0.85   # Slow learner — needs consistent evidence
    MIN_TRADES        = 12
    FULL_BLEND_TRADES = 40
    MAX_BLEND         = 0.35   # global (validated) weights always keep >= 65%
    MAX_MULT          = 2.0
    MIN_MULT          = 0.15

    SIGNALS = ['momentum', 'fii_dii', 'pead', 'mean_rev', 'bulk_deal',
               'delivery_pct', 'option_chain', 'insider', 'fo_ban', 'corp_event',
               'hi52', 'sector_mom', 'pledge', 'sast', 'shp_delta']

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
                cls.MAX_BLEND * (c - cls.MIN_TRADES) /
                max(cls.FULL_BLEND_TRADES - cls.MIN_TRADES, 1),
                cls.MAX_BLEND
            )
            out[s] = round((1 - blend) * 1.0 + blend * learned.get(s, 1.0), 3)
        return out

    @classmethod
    def update_from_outcome(cls, symbol: str,
                            alpha_scores: Dict,
                            fwd_ret: float):
        """
        Update per-stock signal multipliers from one known outcome.
        fwd_ret: the graded net return (5-trading-day horizon preferred —
        matched to what the alphas actually predict; 1d is noise for them).
        alpha_scores: {'signal_name': {'score': x, 'confidence': y}, ...}
        """
        if not alpha_scores or fwd_ret is None or np.isnan(float(fwd_ret)):
            return
        data = cls._load()
        if symbol not in data:
            data[symbol] = {'n_trades': 0, 'multipliers': {}}
        sd = data[symbol]
        sd['n_trades'] = sd.get('n_trades', 0) + 1
        mults = sd.setdefault('multipliers', {})
        counts = sd.setdefault('sig_counts', {})   # per-signal evidence count
        pos_outcome = float(fwd_ret) > 0

        for sig in cls.SIGNALS:
            comp  = alpha_scores.get(sig, {})
            score = comp.get('score', 0.0) if isinstance(comp, dict) else 0.0
            conf  = comp.get('confidence', 0.0) if isinstance(comp, dict) else 0.0
            if conf <= 0 or abs(score) < 0.05:
                continue
            counts[sig] = counts.get(sig, 0) + 1   # this signal had a real view
            agreed = (score > 0) == pos_outcome
            target = 1.25 if agreed else 0.75
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
            # 5d outcome preferred (horizon-matched); 1d only as fallback
            fwd_ret = row.get('ret_5d')
            if pd.isna(fwd_ret):
                fwd_ret = row.get('ret_1d')
            if pd.isna(fwd_ret):
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
                cls.update_from_outcome(sym, alpha_scores, float(fwd_ret))
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
                    'vix': vix_value, 'nifty_trend': 0.0,
                    'trend_gate_open': True, 'nifty_vs_200dma': 0.0,
                }

            close = nifty_df['Close']
            sma20 = close.rolling(20).mean().iloc[-1]
            sma50 = close.rolling(50).mean().iloc[-1]
            nifty_trend = (sma20 / sma50 - 1) if sma50 > 0 else 0.0

            # ── 200-DMA TREND GATE ────────────────────────────────────────────
            # Master long-only filter: when NIFTY closes below its 200-day
            # moving average, no new BUY signals are emitted (cash is a
            # position). The single most reliable drawdown filter for Indian
            # momentum systems — long-biased signals in a downtrend were the
            # main driver of the May-June 2026 underperformance.
            # Fail-open if fewer than 200 sessions of history (a closed gate
            # from MISSING DATA would silently kill the system).
            trend_gate_open  = True
            nifty_vs_200dma  = 0.0
            if len(close) >= 200:
                sma200 = float(close.rolling(200).mean().iloc[-1])
                if sma200 > 0:
                    last_close       = float(close.iloc[-1])
                    nifty_vs_200dma  = (last_close / sma200 - 1) * 100
                    trend_gate_open  = last_close > sma200

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
                'trend_gate_open': trend_gate_open,
                'nifty_vs_200dma': round(nifty_vs_200dma, 2),
            }

        except Exception as e:
            return {
                'regime': 'SIDEWAYS', 'confidence': 20.0,
                'vix': 15.0, 'nifty_trend': 0.0,
                'nifty_3d_change': 0.0,
                'regime_momentum': 'STABLE', 'transition_risk': 0.10,
                'trend_gate_open': True, 'nifty_vs_200dma': 0.0,
            }


# ==============================================================================
# MULTI-ALPHA ENGINE
# ==============================================================================
class MultiAlphaEngine:
    """
    Master engine: fetches all alpha signals, applies regime filter,
    cross-sectional ranking, and risk gates.
    """

    # Regime-specific alpha weights — what works in each regime.
    #
    # [2026-06-19 RE-WEIGHT] The 169-trade graded record exposed that, of the
    # alphas with REAL composite weight, only `momentum` had a live data feed —
    # and its 5d rank-IC this period was ~0 (+0.015). The two alphas that DID
    # carry a measurable edge in the shadow grades — delivery_pct (+0.09) and
    # sector_mom (+0.09) — had been starved of weight. pead/fii_dii are validated
    # but currently produce NO data in the cloud feed (0/169 active), so they
    # contribute nothing until that pipeline is restored. We therefore route the
    # bulk of the weight to the data-having, positive-edge alphas (delivery_pct,
    # sector_mom) while keeping momentum and the validated-but-dark alphas in the
    # map so they resume weight automatically when their data returns.
    # ml_score stays SMALL: walk-forward 2020-2026 (walk_forward_report.md) shows
    # the pooled model has no standalone edge (near-random AUC most years).
    REGIME_WEIGHTS = {
        'BULL': {
            'ml_score':     0.08,
            'momentum':     0.20,  # 12-1 momentum shines in bull markets
            'fii_dii':      0.12,  # Flow momentum confirms trend (dark: no data)
            'pead':         0.12,  # Earnings follow-through (dark: no data)
            'mean_rev':     0.04,  # Less useful in strong trends
            'bulk_deal':    0.05,
            'insider':      0.03,
            'delivery_pct': 0.13,  # institutional accumulation — live + edge
            'sector_mom':   0.11,  # sector rotation — live + edge
            'option_chain': 0.04,
            'corp_event':   0.03,
            'hi52':         0.06,
        },
        'BEAR': {
            'ml_score':     0.06,
            'momentum':     0.04,  # Momentum destroys capital in bear markets
            'fii_dii':      0.15,  # FII selling is the bear driver (dark: no data)
            'pead':         0.06,
            'mean_rev':     0.18,  # Bear market bounces
            'bulk_deal':    0.07,
            'insider':      0.04,
            'delivery_pct': 0.13,  # distribution pattern — live + edge
            'sector_mom':   0.10,  # defensive-sector rotation — live + edge
            'option_chain': 0.05,
            'corp_event':   0.03,
            'hi52':         0.02,
        },
        'SIDEWAYS': {
            'ml_score':     0.06,
            'momentum':     0.10,
            'fii_dii':      0.10,  # dark: no data feed currently
            'pead':         0.13,  # dark: no data feed currently
            'mean_rev':     0.12,  # mean reversion useful but shadow (neg edge)
            'bulk_deal':    0.07,
            'insider':      0.04,
            'delivery_pct': 0.16,  # live + best measured 5d edge → top weight
            'sector_mom':   0.12,  # live + edge
            'option_chain': 0.05,
            'corp_event':   0.03,
            'hi52':         0.05,
        },
        'CRISIS': {
            'ml_score':     0.04,  # ML unreliable in extreme events
            'momentum':     0.04,
            'fii_dii':      0.22,  # FII flows dominate (dark: no data)
            'mean_rev':     0.22,  # Deep oversold bounces
            'pead':         0.04,
            'bulk_deal':    0.07,
            'insider':      0.04,
            'delivery_pct': 0.12,  # panic selling vs institutional buying — live
            'sector_mom':   0.08,  # live
            'option_chain': 0.05,
            'corp_event':   0.03,
        },
    }

    def __init__(
        self,
        data_dir: str = "data/stocks",
        model_cache_dir: str = "model_cache",
        min_composite_score: float = 0.30,  # Min abs score to act
        min_confidence: float = 65.0,       # Min confidence to act
        min_alpha_conviction: int = 2,      # Min live alphas with a real view
    ):
        self.data_dir = data_dir
        self.model_cache_dir = Path(model_cache_dir)
        self.min_composite_score = min_composite_score
        self.min_confidence = min_confidence
        # [2026-06-19] A BUY must be backed by at least this many live alphas that
        # actually hold an opinion (|score|>0.05). In the graded record, 43% of
        # trades fired with ZERO alpha conviction (pure noise) and lost -1.6%.
        self.min_alpha_conviction = min_alpha_conviction

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
            'hi52':         'hi52',
            'sector_mom':   'sector_mom',
            'pledge':       'pledge',
            'sast':         'sast',
            'shp_delta':    'shp_delta',
        }
        for alpha_key, weight_key in alpha_name_map.items():
            comp  = components.get(alpha_key, {})
            score = comp.get('score', 0.0)
            conf  = comp.get('confidence', 0.0)
            # Tier system: shadow alphas (incubating/disabled) carry real
            # scores for grading but get ZERO composite weight until they
            # earn promotion with live evidence (signal_decay_detector).
            is_live = comp.get('live', True)
            if conf > 0 and is_live:
                # [Improvement 4] scale regime weight by per-stock multiplier.
                # Alphas absent from a regime map (e.g. fo_ban) get a small
                # default so they still contribute; normalisation handles sums.
                eff_w           = weights.get(weight_key, 0.04) * stock_mults.get(alpha_key, 1.0)
                weighted_score += score * eff_w
                total_weight   += eff_w
            if conf > 0 and is_live and abs(score) > 0.05:
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

        # (Per-stock performance dampener removed 2026-06-12 — procyclical.
        #  Portfolio-level protection: circuit_breaker.py + 200-DMA gate.)

        # Composite confidence — live signals only
        confs = [c.get('confidence', 0) for c in components.values()
                 if c.get('confidence', 0) > 0 and c.get('live', True)]
        if ml_confidence > 0:
            confs.append(ml_confidence)
        composite_conf = float(np.mean(confs)) if confs else 0.0

        # Number of live alphas that actually hold a directional view.
        n_conviction = len(signals_with_view)

        # Signal determination — LONG-ONLY.
        # The system is structurally long-only (200-DMA gate + circuit breaker;
        # "cash is the alternative"). A negative composite no longer emits a
        # tradeable SELL — it just means "not a buy" (HOLD). Weak names are still
        # surfaced via the cross-sectional AVOID label for visibility/ranking.
        # A BUY additionally requires >= min_alpha_conviction live alphas with a
        # real opinion, so we never trade on a thresholded-noise composite.
        if (composite_score > self.min_composite_score
                and composite_conf >= self.min_confidence
                and n_conviction >= self.min_alpha_conviction):
            signal = 'BUY'
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
            'n_conviction':         n_conviction,
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
        trend_gate_open = regime_info.get('trend_gate_open', True)
        nifty_vs_200dma = regime_info.get('nifty_vs_200dma', 0.0)

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
            gate_str = "OPEN" if trend_gate_open else "CLOSED — no new BUYs"
            print(f"[GATE]   200-DMA trend gate {gate_str} "
                  f"(NIFTY {nifty_vs_200dma:+.1f}% vs 200-DMA)")
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
            raw_conv  = int(df.at[idx, 'n_conviction']) if 'n_conviction' in df.columns else 0
            cur_sig   = df.at[idx, 'signal']
            if cur_sig == 'BUY' and raw_score < adj_min:
                df.at[idx, 'signal'] = 'HOLD'   # score not strong enough after scaling
            elif (cur_sig == 'HOLD' and raw_score > adj_min
                  and raw_conf >= self.min_confidence
                  and raw_conv >= self.min_alpha_conviction):
                df.at[idx, 'signal'] = 'BUY'    # score now qualifies after easing
            # LONG-ONLY: the breadth re-evaluation never creates SELLs.

        # ── Cross-sectional ranking labels ───────────────────────────────────
        df['rank'] = range(1, len(df) + 1)
        df['cs_label'] = 'NEUTRAL'

        # Top N → BUY candidates
        top_mask = df['rank'] <= top_n_long
        df.loc[top_mask & (df['composite_score'] > 0.15), 'cs_label'] = 'TOP_BUY'

        # Bottom N → AVOID
        bottom_mask = df['rank'] > (len(df) - top_n_avoid)
        df.loc[bottom_mask & (df['composite_score'] < -0.15), 'cs_label'] = 'AVOID'

        # Upgrade HOLD → BUY only if score passes the scaled threshold AND the
        # name clears the confidence floor and the minimum-alpha-conviction gate.
        # (A high cross-sectional rank alone must not bypass the quality gates.)
        df.loc[
            (df['cs_label'] == 'TOP_BUY') &
            (df['signal'] == 'HOLD') &
            (df['composite_score'] >= adj_min) &
            (df['composite_confidence'] >= self.min_confidence) &
            (df['n_conviction'] >= self.min_alpha_conviction),
            'signal'
        ] = 'BUY'

        # ── 200-DMA TREND GATE (final word on the long side) ─────────────────
        # NIFTY below its 200-day MA → every BUY becomes HOLD. Long-only
        # alpha in a confirmed downtrend is how the system bled vs the index;
        # holding cash until the gate reopens is itself the position.
        if not trend_gate_open:
            n_gated = int((df['signal'] == 'BUY').sum())
            df.loc[df['signal'] == 'BUY', 'signal'] = 'HOLD'
            if verbose and n_gated:
                print(f"\n[GATE] Trend gate CLOSED: {n_gated} BUY signal(s) "
                      f"downgraded to HOLD (NIFTY {nifty_vs_200dma:+.1f}% "
                      f"below 200-DMA)")

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
        df['trend_gate_open']  = trend_gate_open
        df['nifty_vs_200dma']  = nifty_vs_200dma

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
