"""
Tests for the 2026-06-12 self-correction overhaul:
  - NIFTY 200-DMA trend gate (RegimeDetector)
  - Portfolio circuit breaker (trip / probation / epoch)
  - Alpha incubator tiers (promotion, get_live_alphas)
  - StockWeightLearner overfitting freeze (no learning below 30 outcomes)
"""

import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Trend gate
# ---------------------------------------------------------------------------
def _write_nifty_parquet(data_dir: Path, closes: np.ndarray):
    data_dir.mkdir(parents=True, exist_ok=True)
    idx = pd.bdate_range(end='2026-06-12', periods=len(closes))
    df = pd.DataFrame({'Open': closes, 'High': closes * 1.01,
                       'Low': closes * 0.99, 'Close': closes,
                       'Volume': 1_000_000}, index=idx)
    df.to_parquet(data_dir / "NIFTY50.parquet")


class TestTrendGate:
    def test_gate_closed_below_200dma(self, tmp_path):
        from multi_alpha_engine import RegimeDetector
        # 300 sessions: flat at 100, then a crash to 70 — well below 200-DMA
        closes = np.concatenate([np.full(250, 100.0),
                                 np.linspace(100, 70, 50)])
        _write_nifty_parquet(tmp_path, closes)
        info = RegimeDetector.detect(date(2026, 6, 12), str(tmp_path))
        assert info['trend_gate_open'] is False
        assert info['nifty_vs_200dma'] < 0

    def test_gate_open_above_200dma(self, tmp_path):
        from multi_alpha_engine import RegimeDetector
        # Steady uptrend — price above its 200-DMA
        closes = np.linspace(100, 150, 300)
        _write_nifty_parquet(tmp_path, closes)
        info = RegimeDetector.detect(date(2026, 6, 12), str(tmp_path))
        assert info['trend_gate_open'] is True
        assert info['nifty_vs_200dma'] > 0

    def test_gate_fails_open_without_data(self, tmp_path):
        from multi_alpha_engine import RegimeDetector
        info = RegimeDetector.detect(date(2026, 6, 12), str(tmp_path))
        assert info['trend_gate_open'] is True   # missing data must not gate


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------
def _write_perf_csv(n_trades: int, win_rate: float, excess_daily: float,
                    n_days: int = 12, end: date = date(2026, 6, 11)):
    """Synthetic performance.csv in the cwd."""
    rows = []
    per_day = max(n_trades // n_days, 1)
    wins_needed = int(round(n_trades * win_rate))
    k = 0
    for d in range(n_days):
        sig_date = end - timedelta(days=n_days - 1 - d)
        for _ in range(per_day):
            win = k < wins_needed
            ret = 1.0 if win else -1.0
            rows.append({
                'symbol': f'STOCK{k}.NS', 'signal': 'BUY',
                'signal_date': str(sig_date),
                'ret_1d': excess_daily, 'nifty_1d': 0.0,
                'ret_5d': ret, 'win_5d': int(win),
            })
            k += 1
    Path("paper_trading/results").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv("paper_trading/results/performance.csv",
                              index=False)


class TestCircuitBreaker:
    @pytest.fixture(autouse=True)
    def _tmp_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        yield

    def test_trips_on_low_win_rate(self):
        import circuit_breaker as cb
        _write_perf_csv(n_trades=24, win_rate=0.25, excess_daily=0.0)
        state = cb.evaluate(date(2026, 6, 12))
        assert state['tripped'] is True
        assert 'win rate' in state['reason']

    def test_trips_on_negative_excess(self):
        import circuit_breaker as cb
        # 55% win rate (no win-rate trip) but bleeding -0.5%/day vs NIFTY
        _write_perf_csv(n_trades=24, win_rate=0.55, excess_daily=-0.5)
        state = cb.evaluate(date(2026, 6, 12))
        assert state['tripped'] is True
        assert 'excess' in state['reason']

    def test_no_trip_when_healthy(self):
        import circuit_breaker as cb
        _write_perf_csv(n_trades=24, win_rate=0.60, excess_daily=0.2)
        state = cb.evaluate(date(2026, 6, 12))
        assert state['tripped'] is False

    def test_no_trip_below_min_evidence(self):
        import circuit_breaker as cb
        # Terrible stats but only 10 trades — not enough to trip
        _write_perf_csv(n_trades=10, win_rate=0.10, excess_daily=0.0,
                        n_days=5)
        state = cb.evaluate(date(2026, 6, 12))
        assert state['tripped'] is False

    def test_probation_reset_starts_new_epoch(self):
        import circuit_breaker as cb
        _write_perf_csv(n_trades=24, win_rate=0.25, excess_daily=0.0)
        state = cb.evaluate(date(2026, 6, 12))
        assert state['tripped'] is True
        # Still tripped within probation
        state = cb.evaluate(date(2026, 6, 15))
        assert state['tripped'] is True
        # After probation: reset + fresh epoch (old bad trades don't re-trip)
        state = cb.evaluate(date(2026, 6, 20))
        assert state['tripped'] is False
        assert state['epoch_start'] == '2026-06-20'
        state = cb.evaluate(date(2026, 6, 21))
        assert state['tripped'] is False   # old evidence excluded by epoch


# ---------------------------------------------------------------------------
# Alpha incubator tiers
# ---------------------------------------------------------------------------
class TestAlphaTiers:
    @pytest.fixture(autouse=True)
    def _tmp_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        yield

    def test_core_alphas_live_by_default(self):
        import signal_decay_detector as sdd
        live = sdd.get_live_alphas()
        assert sdd.CORE_ALPHAS <= live
        assert 'mean_rev' not in live        # incubating until promoted

    def test_promotion_requires_evidence(self):
        import signal_decay_detector as sdd
        # Strong but thin evidence: must NOT promote
        sdd.update_tiers({'option_chain': {'win_rate': 70.0, 'n_trades': 10}},
                         {})
        assert 'option_chain' not in sdd.get_live_alphas()
        # Solid evidence: promote
        sdd.update_tiers({'option_chain': {'win_rate': 60.0, 'n_trades': 40}},
                         {})
        assert 'option_chain' in sdd.get_live_alphas()

    def test_kill_switch_demotes_promoted_alpha(self):
        import signal_decay_detector as sdd
        sdd.update_tiers({'option_chain': {'win_rate': 60.0, 'n_trades': 40}},
                         {})
        assert 'option_chain' in sdd.get_live_alphas()
        sdd.update_tiers(
            {'option_chain': {'win_rate': 20.0, 'n_trades': 40}},
            {'option_chain': {'disabled': True}})
        assert 'option_chain' not in sdd.get_live_alphas()

    def test_disabled_core_alpha_not_live(self):
        import signal_decay_detector as sdd
        Path("paper_trading").mkdir(parents=True, exist_ok=True)
        with open("paper_trading/disabled_signals.json", 'w') as f:
            json.dump({'momentum': {'disabled': True}}, f)
        assert 'momentum' not in sdd.get_live_alphas()


# ---------------------------------------------------------------------------
# Weight learner freeze
# ---------------------------------------------------------------------------
class TestWeightLearnerFreeze:
    @pytest.fixture(autouse=True)
    def _tmp_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from multi_alpha_engine import StockWeightLearner
        StockWeightLearner._data = None   # reset class cache
        yield
        StockWeightLearner._data = None

    def test_no_learning_below_30_outcomes(self):
        from multi_alpha_engine import StockWeightLearner as WL
        scores = {'momentum': {'score': 0.8, 'confidence': 70.0}}
        for _ in range(29):                       # 29 < MIN_TRADES=30
            WL.update_from_outcome('TEST.NS', scores, -1.0)
        mults = WL.get_multipliers('TEST.NS')
        assert mults['momentum'] == 1.0           # frozen at neutral

    def test_learned_blend_capped_at_50pct(self):
        from multi_alpha_engine import StockWeightLearner as WL
        scores = {'momentum': {'score': 0.8, 'confidence': 70.0}}
        for _ in range(100):                      # heavy losing evidence
            WL.update_from_outcome('TEST.NS', scores, -1.0)
        mults = WL.get_multipliers('TEST.NS')
        # EMA floor is 0.5; 50% blend with global 1.0 → can't go below 0.75
        assert mults['momentum'] >= 0.75
        assert mults['momentum'] < 1.0            # but it did learn
