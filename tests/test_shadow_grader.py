"""
Tests for shadow_grader.py on synthetic markets where the right answer is
known: a ranking that matches each stock's true drift must grade as an
edge, a random ranking must not, and nothing may use future prices.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

N_STOCKS = 40
N_DAYS = 160
N_SIGNAL_DAYS = 120


def _make_market(tmp_path, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2026-07-01', periods=N_DAYS)
    drift = np.linspace(-0.004, 0.004, N_STOCKS)          # stock i's true edge
    rets = drift + rng.normal(0, 0.01, (N_DAYS, N_STOCKS))
    closes = 100 * np.cumprod(1 + rets, axis=0)
    data = tmp_path / "data" / "stocks"
    data.mkdir(parents=True)
    syms = [f"S{i:02d}.NS" for i in range(N_STOCKS)]
    for i, s in enumerate(syms):
        c = closes[:, i]
        pd.DataFrame({'Open': np.r_[100.0, c[:-1]], 'Close': c},
                     index=days).to_parquet(data / f"{s[:-3]}.parquet")
    nifty = closes.mean(axis=1)
    pd.DataFrame({'Open': np.r_[100.0, nifty[:-1]], 'Close': nifty},
                 index=days).to_parquet(data / "NIFTY50.parquet")
    return days, syms, drift, rng


def _write_signals(tmp_path, days, syms, scores_fn, gate_open=False,
                   alphas=None):
    sig = tmp_path / "signals"
    alpha_dir = tmp_path / "paper_trading" / "alpha_scores"
    sig.mkdir(parents=True, exist_ok=True)
    alpha_dir.mkdir(parents=True, exist_ok=True)
    for k, d in enumerate(days[:N_SIGNAL_DAYS]):
        scores = scores_fn(k)
        ds = d.date().isoformat()
        (sig / f"{ds}_summary.json").write_text(json.dumps({
            'date': ds, 'trend_gate_open': gate_open,
            'full_ranking': [{'symbol': s, 'composite_score': float(v)}
                             for s, v in zip(syms, scores)]}))
        for i, s in enumerate(syms):
            comps = {name: {'score': float(fn(k)[i]), 'confidence': 60.0}
                     for name, fn in (alphas or {}).items()}
            comps['dark'] = {'score': 0.0, 'confidence': 0.0}
            (alpha_dir / f"{s[:-3]}_{ds}.json").write_text(json.dumps(comps))


@pytest.fixture
def cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_true_ranking_grades_as_edge(cwd):
    import shadow_grader as sg
    days, syms, drift, rng = _make_market(cwd)
    _write_signals(cwd, days, syms, lambda k: drift,
                   alphas={'good': lambda k: drift, 'bad': lambda k: -drift})
    r = sg.main([])
    assert r['verdict'] == 'EDGE'
    assert r['shadow_portfolio']['excess_vs_nifty_pct'] > 0
    assert r['top_vs_bottom']['avg_spread_pct'] > 0
    assert r['rank_ic']['composite']['20d']['ic'] > 0.3
    assert r['rank_ic']['good']['20d']['ic'] > 0.3
    assert r['rank_ic']['bad']['20d']['ic'] < -0.3
    assert 'dark' not in r['rank_ic']          # no view -> not graded
    assert (cwd / "paper_trading/results/shadow_report.md").exists()


def test_random_ranking_is_not_an_edge(cwd):
    import shadow_grader as sg
    days, syms, drift, rng = _make_market(cwd, seed=1)
    noise = rng.normal(size=(N_SIGNAL_DAYS, N_STOCKS))
    _write_signals(cwd, days, syms, lambda k: noise[k])
    r = sg.main([])
    assert r['verdict'] != 'EDGE'
    assert abs(r['rank_ic']['composite']['20d']['ic']) < 0.15


def test_closed_gate_means_live_stayed_in_cash(cwd):
    import shadow_grader as sg
    days, syms, drift, rng = _make_market(cwd)
    _write_signals(cwd, days, syms, lambda k: drift, gate_open=False)
    r = sg.main([])
    assert r['window']['gate_open_days'] == 0
    assert r['shadow_portfolio']['live_gated_pct'] == 0.0
    assert r['shadow_portfolio']['top_picks_net_pct'] is not None


def test_forward_return_starts_at_next_open(cwd):
    """A jump on the signal day itself must not be credited to the pick."""
    import shadow_grader as sg
    days = pd.bdate_range('2026-07-01', periods=6)
    C = pd.DataFrame({'A': [100, 150, 150, 150, 150, 150.0],
                      sg.NIFTY: [100.0] * 6}, index=days)
    O = C.shift(1).fillna(100.0)
    exc = sg.forward_excess([days[1]], C, O, hold=2)
    # signalled after day-1's close at 150: entry day-2 open 150 -> 0%
    assert exc.loc[days[1], 'A'] == pytest.approx(0.0)
    exc0 = sg.forward_excess([days[0]], C, O, hold=2)
    # signalled after day-0's close: entry day-1 open 100 -> +50%
    assert exc0.loc[days[0], 'A'] == pytest.approx(0.5)


def test_unmatured_windows_are_excluded(cwd):
    import shadow_grader as sg
    days = pd.bdate_range('2026-07-01', periods=5)
    C = pd.DataFrame({'A': np.arange(100, 105.0), sg.NIFTY: [100.0] * 5},
                     index=days)
    exc = sg.forward_excess(list(days), C, C, hold=3)
    assert list(exc.index) == list(days[:2])   # later windows not finished


def test_no_prices_reports_no_data(cwd):
    import shadow_grader as sg
    days = pd.bdate_range('2026-07-01', periods=3)
    _write_signals(cwd, days, ['X.NS'], lambda k: [0.5])
    r = sg.main([])
    assert r['verdict'] == 'NO_DATA'
