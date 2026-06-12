"""
=============================================================================
DAILY RUNNER  —  The ONE file to run every morning
=============================================================================
Run this at 8:30 AM IST before market opens:

    python daily_runner.py

What it does:
  1. Detects market regime (Bull/Bear/Sideways/Crisis)
  2. Runs ML ensemble on today's features (from cached data)
  3. Runs India-specific alpha signals (FII/DII, bulk deals, PEAD, etc.)
  4. Combines via regime-adaptive multi-alpha engine
  5. Cross-sectionally ranks the universe
  6. Applies portfolio risk gates (correlation, heat, sector)
  7. Outputs today's TOP BUY/SELL signals with stop-loss + targets
  8. Saves to paper_trading/records/signals_YYYY-MM-DD.csv
  9. Tracks prediction outcomes for model improvement

Usage:
    python daily_runner.py                      # Full universe (Nifty 500 subset)
    python daily_runner.py --stocks RELIANCE.NS TCS.NS INFY.NS
    python daily_runner.py --top 10             # Only top 10 signals
    python daily_runner.py --quick              # Skip ML training, alpha only
=============================================================================
"""

import sys
import io
# Windows UTF-8 fix
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except AttributeError:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from pathlib import Path
import json
import traceback

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


# ==============================================================================
# CONFIGURATION
# ==============================================================================
class RunnerConfig:
    # Universe
    NIFTY_50 = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'ICICIBANK.NS',
        'HINDUNILVR.NS', 'ITC.NS', 'SBIN.NS', 'BHARTIARTL.NS', 'KOTAKBANK.NS',
        'LT.NS', 'AXISBANK.NS', 'ASIANPAINT.NS', 'MARUTI.NS', 'TITAN.NS',
        'SUNPHARMA.NS', 'ULTRACEMCO.NS', 'NESTLEIND.NS', 'WIPRO.NS', 'M&M.NS',
        'BAJFINANCE.NS', 'HCLTECH.NS', 'NTPC.NS', 'POWERGRID.NS', 'TATASTEEL.NS',
        'JSWSTEEL.NS', 'ADANIENT.NS', 'ADANIPORTS.NS', 'COALINDIA.NS', 'ONGC.NS',
        'BPCL.NS', 'GRASIM.NS', 'DIVISLAB.NS', 'DRREDDY.NS', 'CIPLA.NS',
        'TECHM.NS', 'APOLLOHOSP.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS', 'BAJAJFINSV.NS',
        'TATACONSUM.NS', 'BRITANNIA.NS', 'HINDALCO.NS', 'INDUSINDBK.NS', 'SBILIFE.NS',
        'HDFCLIFE.NS', 'TATAPOWER.NS', 'UPL.NS', 'LTIM.NS', 'BAJAJ-AUTO.NS'
    ]

    NIFTY_NEXT_50_ADDITIONS = [
        'DMART.NS', 'PIDILITIND.NS', 'TORNTPHARM.NS', 'BERGEPAINT.NS',
        'COLPAL.NS', 'MCDOWELL-N.NS', 'GODREJCP.NS', 'SIEMENS.NS',
        'HAVELLS.NS', 'MARICO.NS', 'DABUR.NS', 'PGHH.NS', 'ABB.NS',
        'BANKBARODA.NS', 'PNB.NS', 'CANBK.NS', 'MUTHOOTFIN.NS',
        'BAJAJHLDNG.NS', 'CHOLAFIN.NS', 'LICI.NS',
    ]

    # Nifty 500 with liquidity floor — the edge lives in liquid mid-caps that
    # funds can't touch. Falls back to the legacy large-cap list if the NSE
    # constituent list is unreachable (see universe.py).
    try:
        from universe import get_universe as _get_universe
        DEFAULT_UNIVERSE = _get_universe()
    except Exception:
        DEFAULT_UNIVERSE = NIFTY_50 + NIFTY_NEXT_50_ADDITIONS

    # Risk
    MAX_POSITIONS = 10
    MAX_PORTFOLIO_HEAT_PCT = 30.0
    MAX_SINGLE_POSITION_PCT = 2.4
    MAX_CORRELATION = 0.70
    MAX_SECTOR_WEIGHT_PCT = 30.0
    KELLY_FRACTION = 0.25

    # Signal quality gates
    MIN_SIGNAL_CONFIDENCE = 60.0  # % — only show signals above this
    MIN_COMPOSITE_SCORE = 0.25    # absolute alpha score required

    # Stop-loss / target (ATR multiples)
    ATR_STOP_MULT = 2.0
    ATR_TARGET_MULT = 3.0

    # Output
    RECORDS_DIR = ROOT / "paper_trading" / "records"
    REPORTS_DIR = ROOT / "daily_reports"
    DATA_DIR = ROOT / "data" / "stocks"
    ML_CACHE_DIR = ROOT / "model_cache"


# ==============================================================================
# ML SIGNAL GENERATOR (calls ModelManager per stock)
# ==============================================================================
class MLSignalGenerator:
    """Runs ML ensemble on today's data and returns per-stock scores."""

    def __init__(self, config: RunnerConfig):
        self.config = config
        self._model_manager = None
        self._data_manager = None
        self._feature_engine = None

    def _init_pipeline(self):
        """Lazy-init ML pipeline (expensive, done once)."""
        if self._data_manager is not None:
            return True
        try:
            from data_manager import DataManager
            from feature_engine import FeatureEngine
            from model_manager import ModelManager
            from data_contracts import validate_and_prepare_data
            self._data_manager = DataManager()
            self._feature_engine = FeatureEngine()
            self._model_manager_class = ModelManager
            self._validate = validate_and_prepare_data
            return True
        except Exception as e:
            print(f"   ⚠️ ML pipeline init failed: {e}")
            return False

    def generate(
        self,
        symbols: list,
        as_of_date: date,
        horizon: int = 5,
        threshold: float = 0.02,
    ) -> dict:
        """
        Returns dict: {symbol: (ml_score float[-1,1], ml_confidence float[0,100])}

        ml_score > 0 → bullish ML view
        ml_score < 0 → bearish ML view
        """
        if not self._init_pipeline():
            return {s: (0.0, 0.0) for s in symbols}

        results = {}
        end_date = as_of_date
        start_date = end_date - timedelta(days=365 * 3)

        for symbol in symbols:
            try:
                # ── Fetch data ──────────────────────────────────────────────
                df = self._data_manager.fetch_stock_data_with_features(
                    symbol, start_date, end_date,
                    compute_features=True, force_refresh=False
                )
                if df is None or len(df) < 250:
                    results[symbol] = (0.0, 0.0)
                    continue

                # ── Validate ─────────────────────────────────────────────────
                try:
                    df, quality_report = self._validate(df, symbol)
                    if not quality_report.is_valid:
                        results[symbol] = (0.0, 0.0)
                        continue
                except Exception:
                    pass  # Proceed without validation if contracts fail

                # ── Feature columns ───────────────────────────────────────────
                exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                           'Target', 'Returns'}
                feature_cols = [c for c in df.columns
                                if c not in exclude
                                and not c.startswith('Future')
                                and df[c].dtype in (np.float64, np.float32,
                                                    np.int64, np.int32)]
                if len(feature_cols) < 5:
                    results[symbol] = (0.0, 0.0)
                    continue

                # ── Training labels ───────────────────────────────────────────
                future_ret = df['Close'].pct_change(horizon).shift(-horizon)
                df['Target'] = np.where(future_ret > threshold, 1,
                               np.where(future_ret < -threshold, 0, np.nan))

                df_clean = df.dropna(subset=['Target']).sort_index()
                if len(df_clean) < 200:
                    results[symbol] = (0.0, 0.0)
                    continue

                X_tr_full = (df_clean[feature_cols]
                             .replace([np.inf, -np.inf], np.nan).fillna(0))
                y_tr_full = df_clean['Target'].astype(int)

                if y_tr_full.nunique() < 2:
                    results[symbol] = (0.0, 0.0)
                    continue

                split = int(len(X_tr_full) * 0.8)
                X_tr = X_tr_full.iloc[:split]
                y_tr = y_tr_full.iloc[:split]
                X_val = X_tr_full.iloc[split:]
                y_val = y_tr_full.iloc[split:]

                # ── Train ─────────────────────────────────────────────────────
                models = self._model_manager_class.train_complete(
                    X_tr, y_tr, X_val=X_val, y_val=y_val,
                    use_moe=False, use_tabnet=False,
                    use_attention=False, use_nas=False, use_flaml=False
                )

                # ── Predict on TODAY (most recent row) ────────────────────────
                X_full = (df[feature_cols]
                          .replace([np.inf, -np.inf], np.nan).fillna(0))
                X_latest = X_full.iloc[[-1]]

                predictions, confidence, proba, info = self._model_manager_class.predict_complete(
                    models, X_latest,
                    use_moe=False, use_tabnet=False, use_conformal=False
                )

                pred = int(predictions[0])
                conf = float(confidence[0]) if isinstance(confidence, np.ndarray) else float(confidence)

                # Normalise: BUY→positive, SELL→negative
                if pred == 1:
                    ml_score = conf  # +conf
                elif pred == 0:
                    ml_score = -conf  # -conf
                else:
                    ml_score = 0.0

                results[symbol] = (float(np.clip(ml_score, -1.0, 1.0)), float(conf * 100))

                # Cache prediction for multi-alpha engine
                nse_sym = symbol.replace('.NS', '').upper()
                try:
                    import joblib
                    cache_path = self.config.ML_CACHE_DIR / f"{nse_sym}_latest_pred.joblib"
                    self.config.ML_CACHE_DIR.mkdir(exist_ok=True)
                    joblib.dump({'date': str(as_of_date), 'score': ml_score,
                                 'confidence': conf * 100}, cache_path)
                except Exception:
                    pass

            except Exception as e:
                results[symbol] = (0.0, 0.0)
                # Uncomment for debug: print(f"   ⚠️ ML error {symbol}: {e}")

        return results


# ==============================================================================
# PORTFOLIO RISK GATE
# ==============================================================================
class PortfolioRiskGate:
    """
    Applies all risk constraints before finalising signals.
    Enforces: correlation, sector, heat, position count, Kelly sizing.
    """

    def __init__(self, config: RunnerConfig):
        self.config = config

    def _get_returns(self, symbol: str, lookback: int = 60) -> pd.Series:
        nse_sym = symbol.replace('.NS', '').upper()
        fpath = self.config.DATA_DIR / f"{nse_sym}.parquet"
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath)
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                return df['Close'].pct_change().dropna().iloc[-lookback:]
            except Exception:
                pass
        return pd.Series(dtype=float)

    def _fetch_yf(self, symbol: str, days: int = 30) -> pd.DataFrame:
        """Fetch recent OHLCV from yfinance as fallback."""
        try:
            import yfinance as yf
            from datetime import timedelta
            end   = date.today()
            start = end - timedelta(days=days)
            df = yf.download(symbol, start=start, end=end,
                             progress=False, auto_adjust=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df if not df.empty else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    def _get_atr(self, symbol: str) -> float:
        nse_sym = symbol.replace('.NS', '').upper()
        fpath = self.config.DATA_DIR / f"{nse_sym}.parquet"
        # Try local parquet first
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath)
                df = df.sort_index()
                val = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
                if not np.isnan(val) and val > 0:
                    return val
            except Exception:
                pass
        # Fallback: yfinance
        try:
            df = self._fetch_yf(symbol, days=60)
            if not df.empty:
                val = float((df['High'] - df['Low']).rolling(14).mean().iloc[-1])
                if not np.isnan(val) and val > 0:
                    return val
        except Exception:
            pass
        return 0.0

    def _get_current_price(self, symbol: str) -> float:
        nse_sym = symbol.replace('.NS', '').upper()
        fpath = self.config.DATA_DIR / f"{nse_sym}.parquet"
        # Try local parquet first
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath)
                val = float(df['Close'].iloc[-1])
                if not np.isnan(val) and val > 0:
                    return val
            except Exception:
                pass
        # Fallback: yfinance
        try:
            df = self._fetch_yf(symbol, days=5)
            if not df.empty:
                val = float(df['Close'].iloc[-1])
                if not np.isnan(val) and val > 0:
                    return val
        except Exception:
            pass
        return 0.0

    def filter_and_size(
        self,
        ranked_df: pd.DataFrame,
        total_capital: float,
        current_positions: list,
        sector_map: dict,
    ) -> list:
        """
        Filter and size the top signals from ranked_df.

        Returns list of trade dicts:
          symbol, signal, confidence, composite_score,
          current_price, stop_loss, target, position_size_inr,
          position_pct, risk_reward, reject_reason
        """
        from PortfolioRiskLimits import PortfolioRiskLimits

        approved = []
        rejected = []
        current_heat = sum(p.get('position_pct', 0) for p in current_positions)
        current_syms = [p['symbol'] for p in current_positions]
        returns_cache = {}

        for _, row in ranked_df.iterrows():
            if row['signal'] not in ('BUY', 'SELL'):
                continue
            if row['composite_confidence'] < self.config.MIN_SIGNAL_CONFIDENCE:
                rejected.append({'symbol': row['symbol'], 'reason': 'low_confidence'})
                continue

            symbol = row['symbol']
            nse_sym = symbol.replace('.NS', '').upper()

            # ── Position count gate ──────────────────────────────────────────
            if len(approved) + len(current_positions) >= self.config.MAX_POSITIONS:
                rejected.append({'symbol': symbol, 'reason': 'max_positions'})
                continue

            # ── Correlation gate ─────────────────────────────────────────────
            if current_syms:
                # Build returns dict lazily
                for s in current_syms + [symbol]:
                    if s not in returns_cache:
                        returns_cache[s] = self._get_returns(s)
                allowed, corr_val, corr_sym = PortfolioRiskLimits.check_correlation(
                    new_symbol=symbol,
                    current_symbols=current_syms,
                    returns_data=returns_cache,
                    lookback=60
                )
                if not allowed:
                    rejected.append({'symbol': symbol, 'reason': f'corr_{corr_val:.2f}_{corr_sym}'})
                    continue

            # ── Sector concentration gate ────────────────────────────────────
            if symbol in sector_map and not PortfolioRiskLimits.check_sector_concentration(
                current_positions=[{'symbol': s, 'kelly_pct': p.get('position_pct', 0)}
                                    for s, p in zip(current_syms, current_positions)],
                new_symbol=nse_sym,
                new_kelly_pct=self.config.MAX_SINGLE_POSITION_PCT,
                sector_map={s.replace('.NS', '').upper(): v for s, v in sector_map.items()}
            ):
                rejected.append({'symbol': symbol, 'reason': 'sector_concentration'})
                continue

            # ── Portfolio heat gate ──────────────────────────────────────────
            kelly_pct = self.config.MAX_SINGLE_POSITION_PCT  # simplified
            scaled = PortfolioRiskLimits.check_portfolio_heat(
                current_positions=[{'kelly_pct': p.get('position_pct', 0)} for p in current_positions]
                                   + [{'kelly_pct': t.get('position_pct', 0)} for t in approved],
                new_position_kelly=kelly_pct
            )
            if scaled <= 0:
                rejected.append({'symbol': symbol, 'reason': 'portfolio_heat'})
                continue

            # ── Price, stop-loss, target ─────────────────────────────────────
            current_price = self._get_current_price(symbol)
            atr = self._get_atr(symbol)

            if not current_price or current_price <= 0 or np.isnan(current_price):
                rejected.append({'symbol': symbol, 'reason': 'no_price'})
                continue

            atr_pct = atr / current_price if current_price > 0 else 0.02
            atr_pct = max(atr_pct, 0.005)  # Floor at 0.5%

            if row['signal'] == 'BUY':
                stop_loss = current_price * (1 - atr_pct * self.config.ATR_STOP_MULT)
                target = current_price * (1 + atr_pct * self.config.ATR_TARGET_MULT)
            else:
                stop_loss = current_price * (1 + atr_pct * self.config.ATR_STOP_MULT)
                target = current_price * (1 - atr_pct * self.config.ATR_TARGET_MULT)

            risk_per_trade = abs(current_price - stop_loss) / current_price
            reward_per_trade = abs(target - current_price) / current_price
            rr_ratio = reward_per_trade / risk_per_trade if risk_per_trade > 0 else 0

            # ── Position sizing (Kelly-scaled by confidence, vol-adjusted) ───
            conf_pct = row['composite_confidence'] / 100.0
            kelly_sized_pct = (self.config.MAX_SINGLE_POSITION_PCT *
                               self.config.KELLY_FRACTION *
                               conf_pct * 4)  # scale back to ~2-3% range
            # Risk parity: a 4%-ATR stock gets half the size of a 2%-ATR
            # stock so each position risks roughly equal capital.
            vol_adj = float(np.clip(0.02 / max(atr_pct, 0.005), 0.5, 1.5))
            kelly_sized_pct = min(kelly_sized_pct * vol_adj,
                                  self.config.MAX_SINGLE_POSITION_PCT)
            position_size_inr = total_capital * kelly_sized_pct / 100.0

            approved.append({
                'symbol': symbol,
                'signal': row['signal'],
                'confidence': round(row['composite_confidence'], 1),
                'composite_score': round(row['composite_score'], 3),
                'current_price': round(current_price, 2),
                'stop_loss': round(stop_loss, 2),
                'target': round(target, 2),
                'risk_reward': round(rr_ratio, 2),
                'position_size_inr': round(position_size_inr, 0),
                'position_pct': round(kelly_sized_pct, 2),
                # Recommended holding window — the validated edges (momentum,
                # PEAD, pooled ML rank) pay off over ~5 trading days; exiting
                # daily paid 0.40% round-trip costs for 1d noise.
                'hold_days': 5,
                'regime': row.get('regime', 'UNKNOWN'),
                'vix': row.get('vix', 0),
                'alpha_breakdown': row.get('alpha_components', {}),
            })
            current_syms.append(symbol)

        return approved, rejected


# ==============================================================================
# MAIN DAILY RUNNER
# ==============================================================================
class DailyRunner:

    def __init__(self, config: RunnerConfig = None):
        self.config = config or RunnerConfig()
        self.config.RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        self.ml_generator = MLSignalGenerator(self.config)
        self.risk_gate = PortfolioRiskGate(self.config)

    def _get_sector_map(self) -> dict:
        """Load sector map from sector_mapping.py."""
        try:
            from sector_mapping import SECTOR_MAP
            return SECTOR_MAP
        except Exception:
            return {}

    def run(
        self,
        symbols: list = None,
        as_of_date: date = None,
        total_capital: float = 1_000_000,
        quick_mode: bool = False,
        top_n: int = 10,
        current_positions: list = None,
    ) -> dict:
        """
        Main daily run.

        Args:
            symbols:           Stock universe (default: Nifty 50 + Next 50)
            as_of_date:        Date to run for (default: today)
            total_capital:     Portfolio capital (INR)
            quick_mode:        Skip ML training, use India alphas only
            top_n:             Number of top signals to show
            current_positions: Existing open positions for risk checks

        Returns:
            Dict with signals, portfolio summary, regime info
        """
        as_of_date = as_of_date or date.today()
        symbols = symbols or self.config.DEFAULT_UNIVERSE
        current_positions = current_positions or []

        print("\n" + "=" * 70)
        print(f"  DAILY TRADING RUNNER  —  {as_of_date}  —  Capital: ₹{total_capital:,.0f}")
        print("=" * 70)

        # ── STEP 1: ML Signals ───────────────────────────────────────────────
        ml_scores = {}
        if not quick_mode:
            print(f"\n[1/4] Running ML ensemble on {len(symbols)} stocks...")
            ml_scores = self.ml_generator.generate(symbols, as_of_date)
            ml_buy = sum(1 for s, c in ml_scores.items() if c[0] > 0.3)
            ml_sell = sum(1 for s, c in ml_scores.items() if c[0] < -0.3)
            print(f"      ML signals: {ml_buy} BUY views, {ml_sell} SELL views")
        else:
            # Quick mode: no per-stock training, but the pooled cross-sectional
            # model (trained weekly, committed to models/) predicts in seconds —
            # so the cloud gets real ML intelligence, not alphas-only.
            print(f"\n[1/4] Quick mode — pooled ML predictor on {len(symbols)} stocks...")
            try:
                from ml_predictor import predict_universe
                ml_scores = predict_universe(
                    symbols, as_of_date,
                    data_dir=str(self.config.DATA_DIR), verbose=True)
                if not any(c > 0 for _, c in ml_scores.values()):
                    ml_scores = {}
            except Exception as e:
                print(f"      Pooled ML unavailable ({e}) — alphas only")
                ml_scores = {}

        # ── STEP 2: Multi-Alpha Ranking ──────────────────────────────────────
        print(f"\n[2/4] Running multi-alpha engine + cross-sectional ranking...")
        from multi_alpha_engine import MultiAlphaEngine
        engine = MultiAlphaEngine(
            data_dir=str(self.config.DATA_DIR),
            model_cache_dir=str(self.config.ML_CACHE_DIR),
            min_composite_score=self.config.MIN_COMPOSITE_SCORE,
            min_confidence=self.config.MIN_SIGNAL_CONFIDENCE,
        )

        ranked_df = engine.rank_universe(
            symbols=symbols,
            as_of_date=as_of_date,
            ml_scores=ml_scores if ml_scores else None,
            top_n_long=min(top_n, 5),
            top_n_avoid=5,
            verbose=True
        )

        # ── Universe snapshot (survivorship-bias fix) ────────────────────────
        # Commit today's tradable universe so future backtests can replay
        # history with the constituents that were ACTUALLY in play, not
        # today's survivors. A few KB per day.
        try:
            udir = Path('signals/universe')
            udir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({'symbol': symbols}).to_csv(
                udir / f"universe_{as_of_date}.csv", index=False)
        except Exception:
            pass

        # ── Portfolio circuit breaker (system-level kill switch) ─────────────
        # Per-alpha kill switches never stopped a bleeding composite. If the
        # trailing portfolio stats breach the trip thresholds, every BUY is
        # downgraded to HOLD; ranking/tracking continue so learning never
        # stops. Auto-resets after probation with a fresh evaluation epoch.
        breaker_tripped = False
        try:
            from circuit_breaker import evaluate as evaluate_breaker
            breaker = evaluate_breaker(as_of_date)
            breaker_tripped = bool(breaker.get('tripped'))
        except Exception as e:
            print(f"      Circuit breaker unavailable ({e}) — continuing")
        if breaker_tripped:
            n_gated = int((ranked_df['signal'] == 'BUY').sum())
            if n_gated:
                ranked_df.loc[ranked_df['signal'] == 'BUY', 'signal'] = 'HOLD'
                print(f"      [BREAKER] {n_gated} BUY signal(s) downgraded "
                      f"to HOLD (portfolio circuit breaker tripped)")

        # ── STEP 3: Risk gate + Position sizing ─────────────────────────────
        print(f"\n[3/4] Applying risk gates...")
        sector_map = self._get_sector_map()
        approved, rejected = self.risk_gate.filter_and_size(
            ranked_df=ranked_df,
            total_capital=total_capital,
            current_positions=current_positions,
            sector_map=sector_map,
        )
        print(f"      Approved: {len(approved)} signals | Rejected: {len(rejected)}")

        # ── STEP 4: Display + Save ───────────────────────────────────────────
        print(f"\n[4/4] Saving results...")
        result = self._format_and_save(approved, rejected, ranked_df, as_of_date, total_capital)

        self._print_final_report(approved, as_of_date, result['regime'])

        return result

    def _format_and_save(self, approved, rejected, ranked_df, as_of_date, total_capital) -> dict:
        """Save results to CSV + JSON."""
        today_str = as_of_date.strftime('%Y-%m-%d')

        # Regime from first row
        regime = ranked_df['regime'].iloc[0] if len(ranked_df) > 0 else 'UNKNOWN'
        vix = ranked_df['vix'].iloc[0] if len(ranked_df) > 0 else 0

        trend_gate_open = bool(ranked_df['trend_gate_open'].iloc[0]) \
            if 'trend_gate_open' in ranked_df.columns and len(ranked_df) else True
        nifty_vs_200dma = float(ranked_df['nifty_vs_200dma'].iloc[0]) \
            if 'nifty_vs_200dma' in ranked_df.columns and len(ranked_df) else 0.0
        breaker_tripped = False
        try:
            from circuit_breaker import is_tripped
            breaker_tripped = is_tripped()
        except Exception:
            pass

        result = {
            'date': today_str,
            'regime': regime,
            'vix': float(vix),
            'trend_gate_open': trend_gate_open,
            'nifty_vs_200dma': nifty_vs_200dma,
            'breaker_tripped': breaker_tripped,
            'total_signals': len(approved),
            'total_capital': total_capital,
            'approved_signals': approved,
            'rejected_count': len(rejected),
            'full_ranking': ranked_df[['symbol', 'signal', 'composite_score',
                                        'composite_confidence', 'cs_label']].to_dict('records'),
        }

        # Save signals CSV
        if approved:
            sdf = pd.DataFrame(approved)
            csv_path = self.config.RECORDS_DIR / f"signals_{today_str}.csv"
            sdf.to_csv(csv_path, index=False)
            print(f"      Signals: {csv_path}")

        # Save JSON report
        json_path = self.config.REPORTS_DIR / f"report_{today_str}.json"
        with open(json_path, 'w') as f:
            # Remove non-serialisable alpha_breakdown
            clean = [{k: v for k, v in s.items() if k != 'alpha_breakdown'}
                     for s in approved]
            json.dump({**result, 'approved_signals': clean}, f, indent=2, default=str)
        print(f"      Report:  {json_path}")

        return result

    def _print_final_report(self, approved, as_of_date, regime):
        """Print clean final report."""
        print("\n" + "=" * 70)
        print(f"  TODAY'S SIGNALS — {as_of_date}  [{regime} MARKET]")
        print("=" * 70)

        if not approved:
            print("  ⚠️  No signals passed all filters today.")
            print("      This is correct — quality over quantity.")
            print("      Tomorrow may have better opportunities.")
            return

        buy_signals = [s for s in approved if s['signal'] == 'BUY']
        sell_signals = [s for s in approved if s['signal'] == 'SELL']

        if buy_signals:
            print(f"\n  BUY SIGNALS ({len(buy_signals)}):")
            print(f"  {'Symbol':<15} {'Price':>8} {'Stop':>8} {'Target':>8} {'R:R':>5} {'Size':>10} {'Conf':>5}")
            print("  " + "-" * 65)
            for s in buy_signals:
                print(f"  {s['symbol']:<15} "
                      f"₹{s['current_price']:>7,.1f} "
                      f"₹{s['stop_loss']:>7,.1f} "
                      f"₹{s['target']:>7,.1f} "
                      f"{s['risk_reward']:>4.1f}x "
                      f"₹{s['position_size_inr']:>9,.0f} "
                      f"{s['confidence']:>4.0f}%")

        if sell_signals:
            print(f"\n  SELL/AVOID SIGNALS ({len(sell_signals)}):")
            for s in sell_signals:
                print(f"  {s['symbol']:<15} "
                      f"₹{s['current_price']:>7,.1f} "
                      f"Conf: {s['confidence']:.0f}%")

        # Risk summary
        total_deployed = sum(s['position_size_inr'] for s in approved)
        total_heat = sum(s['position_pct'] for s in approved)
        print(f"\n  PORTFOLIO HEAT: {total_heat:.1f}% / 30% max")
        print(f"  CAPITAL DEPLOYED: ₹{total_deployed:,.0f}")
        print(f"  POSITIONS TO ADD: {len(approved)}")
        print("\n  IMPORTANT: Use stop-losses. Review at market open.")
        print("=" * 70)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Daily Trading Signal Runner'
    )
    parser.add_argument('--stocks', nargs='+', help='Specific stocks to analyse')
    parser.add_argument('--top', type=int, default=10, help='Max top signals to show')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Portfolio capital (INR)')
    parser.add_argument('--quick', action='store_true', help='Skip ML training, alpha-only mode')
    parser.add_argument('--date', type=str, default=None,
                        help='Run for a specific date (YYYY-MM-DD), default=today')
    args = parser.parse_args()

    run_date = date.fromisoformat(args.date) if args.date else date.today()
    config = RunnerConfig()
    runner = DailyRunner(config)

    result = runner.run(
        symbols=args.stocks or None,
        as_of_date=run_date,
        total_capital=args.capital,
        quick_mode=args.quick,
        top_n=args.top,
    )

    return result


if __name__ == '__main__':
    main()
