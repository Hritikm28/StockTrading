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
