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
  6. Updates the paper portfolio (portfolio_book.py): fills yesterday's
     orders, marks NAV, decides orders for the next open
  7. Saves today's BUY orders to paper_trading/records/signals_YYYY-MM-DD.csv
  8. Tracks prediction outcomes for model improvement

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

from market_calendar import latest_session
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

    # Position count, sizing, rebalancing and the trend exit live in
    # portfolio_book.py (15 names, monthly replacements, rank buffer).

    # Signal quality gates (label the engine's per-stock BUY/HOLD view; the
    # portfolio book selects by rank and does not use them)
    # [2026-06-19] Confidence floor 60 -> 65 and score 0.25 -> 0.30. The graded
    # track record (169 trades) showed composite_score has ~zero rank-correlation
    # with the 5d outcome (rho +0.02), while composite_confidence is monotonically
    # predictive (rho +0.29): high-confidence BUYs were ~breakeven vs -2.3% for
    # low-confidence. So we lean on the confidence gate and tighten the score bar.
    MIN_SIGNAL_CONFIDENCE = 65.0  # % — only show signals above this
    MIN_COMPOSITE_SCORE = 0.30    # absolute alpha score required

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
# MAIN DAILY RUNNER
# ==============================================================================
class DailyRunner:

    def __init__(self, config: RunnerConfig = None):
        self.config = config or RunnerConfig()
        self.config.RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        self.config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        self.ml_generator = MLSignalGenerator(self.config)
        self._sectors = None

    def _sector_of(self, symbol: str):
        """Sector for the portfolio's per-sector cap: the hand-made map
        first, then yfinance's sector from the quality cache for the ~40%
        of the universe the map does not cover. None means uncapped."""
        if self._sectors is None:
            self._sectors = {}
            try:
                from quality_data import load as load_quality
                for sym, rec in load_quality().items():
                    if rec.get('sector'):
                        self._sectors[sym] = rec['sector']
            except Exception:
                pass
            try:
                from sector_mapping import STOCK_SECTOR_MAP
                self._sectors.update(STOCK_SECTOR_MAP)
            except Exception:
                pass
        return self._sectors.get(symbol)

    def run(
        self,
        symbols: list = None,
        as_of_date: date = None,
        total_capital: float = 1_000_000,
        quick_mode: bool = False,
        top_n: int = 10,
    ) -> dict:
        """
        Main daily run.

        Args:
            symbols:           Stock universe (default: Nifty 50 + Next 50)
            as_of_date:        Date to run for (default: today)
            total_capital:     Portfolio capital (INR)
            quick_mode:        Skip ML training, use India alphas only
            top_n:             Number of top signals to show

        Returns:
            Dict with signals, portfolio summary, regime info
        """
        as_of_date = as_of_date or date.today()
        symbols = symbols or self.config.DEFAULT_UNIVERSE

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

        # ── STEP 3: Paper portfolio (fills, NAV, today's orders) ────────────
        print(f"\n[3/4] Updating the paper portfolio...")
        from portfolio_book import run_book
        meta = ranked_df.iloc[0] if len(ranked_df) else {}
        book = run_book(
            ranked_df, as_of_date,
            gate_open=bool(meta.get('trend_gate_open', True)),
            nifty_vs_200dma=meta.get('nifty_vs_200dma'),
            breaker_tripped=breaker_tripped,
            sector_of=self._sector_of,
            data_dir=self.config.DATA_DIR)
        approved = self._buy_orders(book, ranked_df, as_of_date)
        n_sell = sum(1 for o in book['orders'] if o['side'] == 'SELL')
        print(f"      {book['action']}: {len(approved)} buy / {n_sell} sell "
              f"order(s) for the next open | {book['n_holdings']} held | "
              f"NAV ₹{book['nav']:,.0f}")

        # ── STEP 4: Display + Save ───────────────────────────────────────────
        print(f"\n[4/4] Saving results...")
        result = self._format_and_save(approved, [], ranked_df, as_of_date,
                                       total_capital, book)

        self._print_final_report(approved, as_of_date, result['regime'], book)

        return result

    def _buy_orders(self, book: dict, ranked_df: pd.DataFrame,
                    as_of_date: date) -> list:
        """Today's BUY orders in the approved-signal format that
        publish_signals / paper_trade_tracker / email_summary read."""
        rows = {r['symbol']: r for _, r in ranked_df.iterrows()}
        out = []
        for o in book['orders']:
            if o['side'] != 'BUY':
                continue
            r = rows.get(o['symbol'], {})
            # Top-ups (trend restored) carry no rank or score of their own;
            # on 2026-11-06 reading o['score'] crashed the whole run.
            score = o.get('score', r.get('composite_score'))
            rank = o.get('rank', r.get('rank'))
            price = None
            f = self.config.DATA_DIR / f"{o['symbol'].replace('.NS', '')}.parquet"
            try:
                price = float(pd.read_parquet(f)['Close'].iloc[-1])
            except Exception:
                pass
            out.append({
                'symbol': o['symbol'], 'signal': 'BUY',
                'confidence': round(float(r.get('composite_confidence', 0)), 1),
                'composite_score': (round(float(score), 3)
                                    if score is not None else None),
                'rank': int(rank) if rank is not None else None,
                'topup': bool(o.get('topup')),
                'current_price': round(price, 2) if price else None,
                'position_size_inr': o['value'],
                'position_pct': round(o['value'] / book['nav'] * 100, 2)
                                if book['nav'] else 0.0,
                # held until it drops out of the top 30 at a monthly review
                'hold_days': 20,
                'regime': r.get('regime', 'UNKNOWN'), 'vix': r.get('vix', 0),
                'alpha_breakdown': r.get('alpha_components', {}),
            })
        return out

    def _format_and_save(self, approved, rejected, ranked_df, as_of_date,
                         total_capital, book=None) -> dict:
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

        # Alpha data coverage — % of universe where each alpha produced a
        # non-zero signal today. Dead data feeds used to fail SILENTLY
        # (every fetch error returns 0.0/0.0); this makes them loud so a
        # dark alpha shows up in the daily summary/email instead of being
        # discovered months later in the graded record.
        alpha_coverage = {}
        if 'alpha_components' in ranked_df.columns and len(ranked_df):
            comps_col = ranked_df['alpha_components']
            names = set()
            for comps in comps_col:
                if isinstance(comps, dict):
                    names.update(comps.keys())
            n_rows = len(ranked_df)
            for name in sorted(names):
                active = sum(
                    1 for comps in comps_col
                    if isinstance(comps, dict)
                    and comps.get(name, {}).get('confidence', 0) > 0)
                alpha_coverage[name] = round(100.0 * active / n_rows, 1)
            dark = [n for n, pct in alpha_coverage.items() if pct == 0.0]
            if dark:
                print(f"      [DATA WARNING] alphas with ZERO coverage today: "
                      f"{', '.join(dark)}")

        result = {
            'date': today_str,
            'regime': regime,
            'vix': float(vix),
            'alpha_coverage': alpha_coverage,
            'trend_gate_open': trend_gate_open,
            'nifty_vs_200dma': nifty_vs_200dma,
            'breaker_tripped': breaker_tripped,
            'total_signals': len(approved),
            'total_capital': total_capital,
            'approved_signals': approved,
            'rejected_count': len(rejected),
            'portfolio': book or {},
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

    def _print_final_report(self, approved, as_of_date, regime, book):
        """Print clean final report."""
        print("\n" + "=" * 70)
        print(f"  PORTFOLIO — {as_of_date}  [{regime} MARKET]  {book['action']}")
        print("=" * 70)
        print(f"  NAV ₹{book['nav']:,.0f} | since {book['inception']}: "
              f"{book['return_pct']:+.2f}% vs NIFTY "
              f"{(book['nifty_return_pct'] or 0):+.2f}% | "
              f"{book['n_holdings']}/{book['target_holdings']} held, "
              f"{book['cash_pct']:.0f}% cash")

        sells = [o for o in book['orders'] if o['side'] == 'SELL']
        if approved:
            print(f"\n  BUY at next open ({len(approved)}):")
            for s in approved:
                what = ('top-up' if s.get('topup')
                        else f"rank {s['rank'] if s['rank'] is not None else '-':>3}")
                print(f"  {s['symbol']:<15} {what:<9} "
                      f"₹{s['position_size_inr']:>10,.0f}")
        if sells:
            print(f"\n  SELL at next open ({len(sells)}):")
            for o in sells:
                print(f"  {o['symbol']:<15} ({o['reason']})")
        if not approved and not sells:
            print("\n  No orders today.")
        print("=" * 70)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Daily Trading Signal Runner'
    )
    parser.add_argument('--stocks', nargs='+', help='Specific stocks to analyse')
    parser.add_argument('--top', type=int, default=5, help='Max top signals to show')
    parser.add_argument('--capital', type=float, default=1_000_000, help='Portfolio capital (INR)')
    parser.add_argument('--quick', action='store_true', help='Skip ML training, alpha-only mode')
    parser.add_argument('--date', type=str, default=None,
                        help='Run for a specific date (YYYY-MM-DD), default=today')
    args = parser.parse_args()

    # The session the data belongs to, not the runner's calendar date: the
    # nightly job starts after midnight IST and can be queued into Saturday.
    run_date = (date.fromisoformat(args.date) if args.date
                else latest_session())
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
