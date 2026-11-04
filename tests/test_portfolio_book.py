"""
Tests for portfolio_book.py: the paper portfolio must trade rarely
(monthly replacements with a rank buffer), fill at the next session's open,
respect the sector cap and the trend exit, and be safe to re-run.
"""

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import portfolio_book as pb

DAYS = pd.bdate_range('2026-11-02', '2026-12-31')
SYMS = [f"S{i:02d}.NS" for i in range(40)]


@pytest.fixture
def market(tmp_path):
    data = tmp_path / "stocks"
    data.mkdir()
    for i, s in enumerate(SYMS + ['NIFTY50']):
        close = np.full(len(DAYS), 100.0 + i)
        opn = close - 1.0                      # opens 1 below the close
        pd.DataFrame({'Open': opn, 'Close': close}, index=DAYS).to_parquet(
            data / f"{s.replace('.NS', '')}.parquet")
    return data


def ranking(order):
    """ranked frame: first symbol in `order` scores highest."""
    n = len(order)
    return pd.DataFrame({'symbol': order,
                         'composite_score': np.linspace(0.9, 0.05, n)})


def step(market, tmp_path, day, order=SYMS, gate=True, vs200=1.0,
         sectors=None, breaker=False):
    sectors = sectors or {}
    return pb.run_book(ranking(order), day, gate_open=gate,
                       nifty_vs_200dma=vs200, breaker_tripped=breaker,
                       sector_of=lambda s: sectors.get(s), data_dir=market,
                       path=tmp_path / "book.json")


def d(s):
    return date.fromisoformat(s)


def test_enters_from_cash_and_fills_at_next_open(market, tmp_path):
    r = step(market, tmp_path, d('2026-11-03'))
    assert r['action'] == 'BUY' and r['status'] == 'CASH'
    assert len(r['orders']) == pb.TARGET_HOLDINGS
    assert [o['symbol'] for o in r['orders']] == SYMS[:pb.TARGET_HOLDINGS]
    r = step(market, tmp_path, d('2026-11-04'))
    assert r['status'] == 'INVESTED' and r['n_holdings'] == pb.TARGET_HOLDINGS
    s00 = next(f for f in r['fills'] if f['symbol'] == 'S00.NS')
    assert s00['price'] == 99.0                 # 04-Nov open, not a close
    book = pb.load_book(d('2026-11-04'), tmp_path / "book.json")
    spent = sum(f['value'] for f in r['fills'])
    assert book['cash'] == pytest.approx(
        pb.START_CAPITAL - spent * (1 + pb.COST_PER_SIDE_PCT / 100), abs=1.0)
    assert r['cash_pct'] < 1.0


def test_no_churn_inside_the_month(market, tmp_path):
    step(market, tmp_path, d('2026-11-03'))
    step(market, tmp_path, d('2026-11-04'))
    r = step(market, tmp_path, d('2026-11-05'), order=SYMS[::-1])
    assert r['orders'] == [] and r['action'] == 'HOLD'


def test_monthly_rebalance_keeps_buffer_and_sells_dropouts(market, tmp_path):
    step(market, tmp_path, d('2026-11-03'))
    step(market, tmp_path, d('2026-11-04'))
    # December: S00 falls to rank 40 (out), S01 to rank 25 (inside buffer)
    order = [s for s in SYMS if s not in ('S00.NS', 'S01.NS')]
    order.insert(24, 'S01.NS')
    order.append('S00.NS')
    r = step(market, tmp_path, d('2026-12-01'), order=order)
    assert r['action'] == 'REBALANCE'
    sells = [o['symbol'] for o in r['orders'] if o['side'] == 'SELL']
    buys = [o['symbol'] for o in r['orders'] if o['side'] == 'BUY']
    assert sells == ['S00.NS']
    assert buys == ['S15.NS']                   # best name not already held
    r = step(market, tmp_path, d('2026-12-02'), order=order)
    assert r['n_holdings'] == pb.TARGET_HOLDINGS
    held = {h['symbol'] for h in r['holdings']}
    assert 'S01.NS' in held and 'S00.NS' not in held


def test_all_out_mode_sells_everything(market, tmp_path, monkeypatch):
    monkeypatch.setattr(pb, 'LOW_EXPOSURE', 0.0)
    step(market, tmp_path, d('2026-11-03'))
    step(market, tmp_path, d('2026-11-04'))
    r = step(market, tmp_path, d('2026-11-05'), gate=False, vs200=-3.0)
    assert r['action'] == 'TREND_EXIT'
    assert all(o['side'] == 'SELL' for o in r['orders'])
    assert len(r['orders']) == pb.TARGET_HOLDINGS
    r = step(market, tmp_path, d('2026-11-06'), gate=False, vs200=-3.0)
    assert r['status'] == 'CASH' and r['n_holdings'] == 0


def test_all_out_mode_inside_band_holds_but_never_buys(market, tmp_path,
                                                       monkeypatch):
    monkeypatch.setattr(pb, 'LOW_EXPOSURE', 0.0)
    r = step(market, tmp_path, d('2026-11-03'), gate=False, vs200=-1.0)
    assert r['action'] == 'CASH' and r['orders'] == []
    step(market, tmp_path, d('2026-11-04'))
    step(market, tmp_path, d('2026-11-05'))
    r = step(market, tmp_path, d('2026-11-06'), gate=False, vs200=-1.0)
    assert r['action'] == 'HOLD' and r['orders'] == []


def test_breaker_blocks_new_buys(market, tmp_path):
    r = step(market, tmp_path, d('2026-11-03'), breaker=True)
    assert r['orders'] == [] and r['action'] == 'CASH'


def test_sector_cap(market, tmp_path):
    sectors = {s: 'Bank' for s in SYMS[:10]}
    r = step(market, tmp_path, d('2026-11-03'), sectors=sectors)
    banks = [o for o in r['orders'] if sectors.get(o['symbol']) == 'Bank']
    assert len(banks) == pb.MAX_PER_SECTOR
    assert len(r['orders']) == pb.TARGET_HOLDINGS


def test_negative_scores_are_never_bought(market, tmp_path):
    ranked = ranking(SYMS)
    ranked.loc[3:, 'composite_score'] = -0.2
    r = pb.run_book(ranked, d('2026-11-03'), True, 1.0, False,
                    lambda s: None, market, tmp_path / "book.json")
    assert [o['symbol'] for o in r['orders']] == SYMS[:3]


def test_rerun_same_day_is_idempotent(market, tmp_path):
    step(market, tmp_path, d('2026-11-03'))
    step(market, tmp_path, d('2026-11-04'))
    r1 = step(market, tmp_path, d('2026-11-05'))
    r2 = step(market, tmp_path, d('2026-11-05'))
    book = pb.load_book(d('2026-11-05'), tmp_path / "book.json")
    assert r1['nav'] == r2['nav']
    assert [p['date'] for p in book['nav']].count('2026-11-05') == 1
    assert len(book['trades']) == pb.TARGET_HOLDINGS


def test_unfillable_orders_expire(market, tmp_path):
    ranked = pd.DataFrame({'symbol': ['GHOST.NS'] + SYMS,
                           'composite_score': [0.99] + [0.5] * len(SYMS)})
    pb.run_book(ranked, d('2026-11-03'), True, 1.0, False, lambda s: None,
                market, tmp_path / "book.json")
    book = pb.load_book(d('2026-11-03'), tmp_path / "book.json")
    assert 'GHOST.NS' not in {o['symbol'] for o in book['pending']}


def test_trend_break_cuts_every_holding_to_half_then_restores(market, tmp_path):
    step(market, tmp_path, d('2026-11-03'))
    full = step(market, tmp_path, d('2026-11-04'))
    shares = {h['symbol']: h for h in full['holdings']}
    r = step(market, tmp_path, d('2026-11-05'), gate=False, vs200=-3.0)
    assert r['action'] == 'TREND_CUT' and r['exposure_pct'] == 50
    assert len(r['orders']) == pb.TARGET_HOLDINGS
    assert all(o['side'] == 'SELL' and o['fraction'] == 0.5 for o in r['orders'])
    r = step(market, tmp_path, d('2026-11-06'), gate=False, vs200=-3.0)
    assert r['n_holdings'] == pb.TARGET_HOLDINGS          # still holds all 15
    assert 45 < r['cash_pct'] < 55                        # about half in cash
    r = step(market, tmp_path, d('2026-11-09'), gate=False, vs200=-1.0)
    assert r['orders'] == [] and r['exposure_pct'] == 50  # inside band: wait
    r = step(market, tmp_path, d('2026-11-10'), vs200=1.0)
    assert r['action'] == 'TREND_RESTORED' and r['exposure_pct'] == 100
    assert all(o['side'] == 'BUY' and o['topup'] for o in r['orders'])
    r = step(market, tmp_path, d('2026-11-11'), vs200=1.0)
    assert r['cash_pct'] < 2 and r['n_holdings'] == pb.TARGET_HOLDINGS


def test_half_mode_enters_at_half_size_while_trend_is_down(market, tmp_path):
    r = step(market, tmp_path, d('2026-11-03'), gate=False, vs200=-1.0)
    assert r['action'] == 'BUY' and r['exposure_pct'] == 50
    assert len(r['orders']) == pb.TARGET_HOLDINGS
    per = pb.START_CAPITAL / pb.TARGET_HOLDINGS
    assert all(o['value'] == pytest.approx(per / 2, rel=1e-3) for o in r['orders'])
    r = step(market, tmp_path, d('2026-11-04'), gate=False, vs200=-1.0)
    assert 45 < r['cash_pct'] < 55


def test_partial_sell_keeps_cost_basis_proportional(market, tmp_path):
    step(market, tmp_path, d('2026-11-03'))
    step(market, tmp_path, d('2026-11-04'))
    before = pb.load_book(d('2026-11-04'), tmp_path / "book.json")['holdings']['S00.NS']
    step(market, tmp_path, d('2026-11-05'), gate=False, vs200=-3.0)
    step(market, tmp_path, d('2026-11-06'), gate=False, vs200=-3.0)
    after = pb.load_book(d('2026-11-06'), tmp_path / "book.json")['holdings']['S00.NS']
    assert after['shares'] == pytest.approx(before['shares'] / 2)
    assert after['cost_basis'] == pytest.approx(before['cost_basis'] / 2)
