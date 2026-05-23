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

            # Regime classification
            if vix_value > 25 or vix_5d_change > 0.25:
                regime = 'CRISIS'
                confidence = min(vix_value * 3, 95.0)
            elif nifty_trend > 0.015 and vix_value < 18:
                regime = 'BULL'
                confidence = min(abs(nifty_trend) * 1000, 85.0)
            elif nifty_trend < -0.015:
                regime = 'BEAR'
                confidence = min(abs(nifty_trend) * 1000, 80.0)
            else:
                regime = 'SIDEWAYS'
                confidence = 60.0

            return {
                'regime': regime,
                'confidence': round(confidence, 1),
                'vix': round(vix_value, 2),
                'nifty_trend': round(nifty_trend * 100, 2)  # in %
            }

        except Exception as e:
            print(f"   ⚠️ RegimeDetector error: {e}")
            return {'regime': 'SIDEWAYS', 'confidence': 20.0, 'vix': 15.0, 'nifty_trend': 0.0}


# ==============================================================================
# MULTI-ALPHA ENGINE
# ==============================================================================
class MultiAlphaEngine:
    """
    Master engine: fetches all alpha signals, applies regime filter,
    cross-sectional ranking, and risk gates.
    """

    # Regime-specific alpha weights — what works in each regime
    REGIME_WEIGHTS = {
        'BULL': {
            'ml_score':    0.30,  # ML trend-following dominant
            'momentum':    0.25,  # 12-1 momentum shines in bull markets
            'fii_dii':     0.15,  # Flow momentum confirms trend
            'pead':        0.15,  # Earnings beat follow-through
            'mean_rev':    0.05,  # Less useful in strong trends
            'bulk_deal':   0.06,
            'insider':     0.04,
        },
        'BEAR': {
            'ml_score':    0.25,
            'momentum':    0.05,  # Momentum destroys capital in bear markets
            'fii_dii':     0.20,  # FII selling is the bear driver
            'pead':        0.10,
            'mean_rev':    0.25,  # Bear market bounces — mean rev is key
            'bulk_deal':   0.10,
            'insider':     0.05,
        },
        'SIDEWAYS': {
            'ml_score':    0.25,
            'momentum':    0.10,
            'fii_dii':     0.12,
            'pead':        0.18,
            'mean_rev':    0.20,  # Mean reversion dominant in sideways
            'bulk_deal':   0.10,
            'insider':     0.05,
        },
        'CRISIS': {
            'ml_score':    0.15,  # ML unreliable in extreme events
            'momentum':    0.05,
            'fii_dii':     0.30,  # FII flows are the dominant factor
            'pead':        0.05,
            'mean_rev':    0.30,  # Deep oversold bounces
            'bulk_deal':   0.10,
            'insider':     0.05,
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

        # ── Combine with regime weights ──────────────────────────────────────
        weighted_score = 0.0
        total_weight = 0.0

        # ML score contribution
        if ml_confidence > 0:
            ml_normalised = np.clip(ml_score, -1.0, 1.0)
            weighted_score += ml_normalised * weights['ml_score']
            total_weight += weights['ml_score']

        # India alpha contributions
        alpha_name_map = {
            'momentum': 'momentum',
            'fii_dii':  'fii_dii',
            'pead':     'pead',
            'mean_rev': 'mean_rev',
            'bulk_deal':'bulk_deal',
            'insider':  'insider',
        }
        for alpha_key, weight_key in alpha_name_map.items():
            comp = components.get(alpha_key, {})
            score = comp.get('score', 0.0)
            conf  = comp.get('confidence', 0.0)
            if conf > 0:
                weighted_score += score * weights[weight_key]
                total_weight += weights[weight_key]

        # F&O ban — always included if confidence > 0
        fo = components.get('fo_ban', {})
        if fo.get('confidence', 0) > 0:
            weighted_score += fo.get('score', 0.0) * 0.07
            total_weight += 0.07

        # Normalise
        composite_score = weighted_score / total_weight if total_weight > 0 else 0.0
        composite_score = float(np.clip(composite_score, -1.0, 1.0))

        # Composite confidence
        confs = [c.get('confidence', 0) for c in components.values() if c.get('confidence', 0) > 0]
        if ml_confidence > 0:
            confs.append(ml_confidence)
        composite_conf = float(np.mean(confs)) if confs else 0.0

        # Signal
        if composite_score > self.min_composite_score and composite_conf >= self.min_confidence:
            signal = 'BUY'
        elif composite_score < -self.min_composite_score and composite_conf >= self.min_confidence:
            signal = 'SELL'
        else:
            signal = 'HOLD'

        # CRISIS override: never BUY in crisis unless very strong
        if regime == 'CRISIS' and composite_score < 0.6:
            signal = 'HOLD'

        return {
            'symbol': symbol,
            'date': as_of_date.isoformat(),
            'signal': signal,
            'composite_score': round(composite_score, 3),
            'composite_confidence': round(composite_conf, 1),
            'regime': regime,
            'ml_score': round(ml_score, 3),
            'ml_confidence': round(ml_confidence, 1),
            'alpha_components': components,
            'weight_used': round(total_weight, 3),
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
        regime_info = RegimeDetector.detect(as_of_date, self.data_dir)
        regime = regime_info['regime']
        regime_conf = regime_info['confidence']

        if verbose:
            print(f"\n{'='*60}")
            print(f"🌐 Market Regime: {regime} ({regime_conf:.0f}% confidence)")
            print(f"   India VIX: {regime_info['vix']:.1f} | "
                  f"Nifty Trend: {regime_info['nifty_trend']:+.2f}%")
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

        # ── Cross-sectional ranking labels ───────────────────────────────────
        df['rank'] = range(1, len(df) + 1)
        df['cs_label'] = 'NEUTRAL'

        # Top N → BUY candidates (override individual HOLD if ranked highly)
        top_mask = df['rank'] <= top_n_long
        df.loc[top_mask & (df['composite_score'] > 0.15), 'cs_label'] = 'TOP_BUY'

        # Bottom N → AVOID
        bottom_mask = df['rank'] > (len(df) - top_n_avoid)
        df.loc[bottom_mask & (df['composite_score'] < -0.15), 'cs_label'] = 'AVOID'

        # Upgrade signals for top ranked
        df.loc[(df['cs_label'] == 'TOP_BUY') & (df['signal'] == 'HOLD'), 'signal'] = 'BUY'

        # Add regime info
        df['regime'] = regime
        df['regime_confidence'] = regime_conf
        df['vix'] = regime_info['vix']

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
