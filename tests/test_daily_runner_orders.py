"""daily_runner._buy_orders must format every kind of BUY order. A top-up
(trend restored) has no rank or score, and on 2026-11-06 it crashed the run."""

import sys
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

import daily_runner


def test_topup_and_new_buy_orders_are_both_formatted(tmp_path):
    pd.DataFrame({'Close': [100.0]}, index=[pd.Timestamp('2026-11-06')]
                 ).to_parquet(tmp_path / "AAA.parquet")
    ranked = pd.DataFrame({'symbol': ['AAA.NS', 'BBB.NS'],
                           'composite_score': [0.9, 0.8],
                           'composite_confidence': [70.0, 60.0],
                           'rank': [1, 2], 'regime': 'BULL', 'vix': 15.0})
    book = {'nav': 1_000_000, 'orders': [
        {'symbol': 'AAA.NS', 'side': 'BUY', 'reason': 'trend_restored',
         'topup': True, 'value': 30000.0},
        {'symbol': 'BBB.NS', 'side': 'BUY', 'reason': 'top_rank',
         'value': 60000.0, 'rank': 2, 'score': 0.8},
        {'symbol': 'CCC.NS', 'side': 'SELL', 'reason': 'rank_dropped'}]}
    fake = SimpleNamespace(config=SimpleNamespace(DATA_DIR=tmp_path))
    out = daily_runner.DailyRunner._buy_orders(fake, book, ranked,
                                               date(2026, 11, 6))
    assert [o['symbol'] for o in out] == ['AAA.NS', 'BBB.NS']
    assert out[0]['topup'] and out[0]['rank'] == 1
    assert out[0]['composite_score'] == 0.9 and out[0]['current_price'] == 100.0
    assert not out[1]['topup'] and out[1]['rank'] == 2


def test_sector_cap_uses_nse_industry_labels_first(monkeypatch):
    """Five of the first fifteen picks had no sector label, so the cap of
    four let six capital-goods companies through."""
    import universe
    import quality_data
    monkeypatch.setattr(universe, 'fetch_nifty500_list', lambda force=False: pd.DataFrame({
        'Symbol': ['BHEL', 'TDPOWERSYS', 'WELCORP'],
        'Industry': ['Capital Goods', 'Capital Goods', 'Capital Goods']}))
    monkeypatch.setattr(quality_data, 'load', lambda: {
        'BHEL.NS': {'sector': 'Industrials'},          # another vocabulary
        'OTHER.NS': {'sector': 'Energy'}})             # not in the NSE list
    fake = SimpleNamespace(_sectors=None)
    sector = lambda s: daily_runner.DailyRunner._sector_of(fake, s)
    assert sector('BHEL.NS') == 'Capital Goods'        # NSE label wins
    assert sector('TDPOWERSYS.NS') == 'Capital Goods'  # was unlabelled: uncapped
    assert sector('WELCORP.NS') == 'Capital Goods'
    assert sector('OTHER.NS') == 'Energy'              # fallback still works
    assert sector('UNKNOWN.NS') is None


def test_sector_cap_falls_back_when_the_nse_list_is_unavailable(monkeypatch):
    import universe
    import quality_data
    monkeypatch.setattr(universe, 'fetch_nifty500_list', lambda force=False: None)
    monkeypatch.setattr(quality_data, 'load', lambda: {'ZZTEST.NS': {'sector': 'Industrials'}})
    fake = SimpleNamespace(_sectors=None)
    assert daily_runner.DailyRunner._sector_of(fake, 'ZZTEST.NS') == 'Industrials'
