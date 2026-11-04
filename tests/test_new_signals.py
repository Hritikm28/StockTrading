"""
Tests for the two probationary signals: earnings surprise (PEADAlpha reading
the yfinance earnings_dates layout that earnings_calendar.py caches) and
quality (QualityAlpha + the quality_data cache).
"""

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Earnings surprise
# ---------------------------------------------------------------------------
def _earnings_cache(tmp_path, rows):
    """rows: [(date, estimate, reported)] in yfinance's column layout."""
    idx = pd.DatetimeIndex([pd.Timestamp(d, tz='Asia/Kolkata') for d, _, _ in rows],
                           name='Earnings Date')
    df = pd.DataFrame({'EPS Estimate': [e for _, e, _ in rows],
                       'Reported EPS': [r for _, _, r in rows]}, index=idx)
    df['Surprise(%)'] = (df['Reported EPS'] - df['EPS Estimate']) / df['EPS Estimate'] * 100
    d = tmp_path / "earnings"
    d.mkdir(exist_ok=True)
    df.to_parquet(d / "ABC_earnings.parquet")
    return d


@pytest.fixture
def pead(tmp_path, monkeypatch):
    from india_alpha_signals import PEADAlpha
    monkeypatch.setattr(PEADAlpha, '_earnings_dir', tmp_path / "earnings")
    return PEADAlpha


def test_reads_yfinance_columns_after_a_beat(tmp_path, pead):
    _earnings_cache(tmp_path, [('2026-07-20', 10.0, 10.2),
                               ('2026-10-20', 10.0, 11.5)])   # +15% beat
    score, conf = pead.get_signal('ABC.NS', date(2026, 10, 30))
    assert score > 0.8 and conf > 0


def test_a_miss_scores_negative(tmp_path, pead):
    _earnings_cache(tmp_path, [('2026-10-20', 10.0, 8.0)])     # -20% miss
    score, _ = pead.get_signal('ABC.NS', date(2026, 10, 25))
    assert score < -0.5


def test_drift_fades_and_ends(tmp_path, pead):
    _earnings_cache(tmp_path, [('2026-09-01', 10.0, 11.5)])
    early, _ = pead.get_signal('ABC.NS', date(2026, 9, 5))
    late, _ = pead.get_signal('ABC.NS', date(2026, 10, 20))
    gone, conf = pead.get_signal('ABC.NS', date(2026, 11, 15))
    assert early > late > 0 and gone == 0 and conf == 0


def test_future_results_are_not_used(tmp_path, pead):
    _earnings_cache(tmp_path, [('2026-11-20', 10.0, 13.0)])
    assert pead.get_signal('ABC.NS', date(2026, 11, 1)) == (0.0, 0.0)


def test_surprise_falls_back_to_yfinance_column():
    from india_alpha_signals import PEADAlpha
    row = pd.Series({'Reported EPS': None, 'EPS Estimate': None,
                     'Surprise(%)': 12.5})
    assert PEADAlpha._surprise_pct(row) == 12.5
    assert PEADAlpha._surprise_pct(pd.Series({'x': 1})) is None


# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------
def _universe(n=30):
    data = {f"S{i:02d}.NS": {'roe': 0.01 * i, 'roa': None, 'de': 100 - i,
                             'pm': 0.1, 'sector': 'Industrials'}
            for i in range(n)}
    return data


def test_quality_ranks_profitable_low_debt_highest():
    from india_alpha_signals import QualityAlpha
    scores = QualityAlpha.build(_universe())
    best, worst = scores['S29.NS'], scores['S00.NS']
    assert best[0] > 0.8 and worst[0] < -0.8
    assert best[1] == 60.0


def test_quality_ignores_debt_for_financials_and_uses_roa_fallback():
    from india_alpha_signals import QualityAlpha
    data = _universe()
    data['BANK.NS'] = {'roe': 0.30, 'roa': None, 'de': 900.0,
                       'sector': 'Financial Services'}
    data['NOROE.NS'] = {'roe': None, 'roa': 0.2, 'de': None, 'sector': None}
    scores = QualityAlpha.build(data)
    assert scores['BANK.NS'][1] == 45.0            # ROE only, debt ignored
    assert scores['BANK.NS'][0] > 0.8
    assert 'NOROE.NS' in scores


def test_quality_needs_a_real_universe():
    from india_alpha_signals import QualityAlpha
    assert QualityAlpha.build(_universe(5)) == {}


def test_quality_cache_refresh_cap_and_retry(tmp_path):
    import quality_data as qd
    path = tmp_path / "quality.json"
    calls = []

    def fetch(sym):
        calls.append(sym)
        return None if sym == 'BAD.NS' else {'roe': 0.2, 'roa': 0.1,
                                             'de': 10.0, 'pm': 0.1,
                                             'sector': 'Technology'}
    syms = ['A.NS', 'B.NS', 'C.NS', 'BAD.NS']
    qd.update(syms, fetch, date(2026, 11, 1), path, max_fetches=3, pause=0)
    assert len(calls) == 3
    qd.update(syms, fetch, date(2026, 11, 2), path, max_fetches=3, pause=0)
    cache = qd.load(path)
    assert set(cache) == set(syms) and cache['BAD.NS']['missing']
    calls.clear()
    qd.update(syms, fetch, date(2026, 11, 5), path, max_fetches=3, pause=0)
    assert calls == []                              # nothing due yet
    qd.update(syms, fetch, date(2026, 11, 10), path, max_fetches=3, pause=0)
    assert calls == ['BAD.NS']                      # weekly retry only
    calls.clear()
    qd.update(syms, fetch, date(2026, 12, 5), path, max_fetches=3, pause=0)
    assert len(calls) == 3                          # monthly refresh, capped
