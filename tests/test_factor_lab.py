"""
Tests for factor_lab.py on synthetic markets: a ranking that knows each
stock's true drift must beat the index, the trend gate must move the book
to cash in a falling market, illiquid stocks must never be bought, costs
must be charged, and the deflated Sharpe must tell signal from noise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import factor_lab as fl
import factors as fx

N_DAYS = 900
N_STOCKS = 60


def make_market(seed=0, nifty_drift=0.0004, crash_at=None):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range('2020-01-01', periods=N_DAYS)
    drift = np.linspace(-0.0008, 0.0012, N_STOCKS)
    rets = drift + rng.normal(0, 0.012, (N_DAYS, N_STOCKS))
    if crash_at is not None:
        rets[crash_at:crash_at + 60] -= 0.006
    close = pd.DataFrame(100 * np.cumprod(1 + rets, axis=0), index=days,
                         columns=[f"S{i:02d}" for i in range(N_STOCKS)])
    opn = close.shift(1).fillna(100.0) * 1.001
    vol = pd.DataFrame(1e6, index=days, columns=close.columns)
    vol['S59'] = 1.0                          # best drift, but illiquid
    nr = nifty_drift + rng.normal(0, 0.002, N_DAYS)   # near-deterministic index
    if crash_at is not None:
        nr[crash_at:crash_at + 60] -= 0.006
    nifty = pd.Series(10000 * np.cumprod(1 + nr), index=days)
    truth = dict(zip(range(N_STOCKS), drift))
    return {'Open': opn, 'Close': close, 'Volume': vol}, nifty, truth


def oracle(truth):
    return lambda closes: {i: truth[i] for i in closes}


def test_oracle_ranking_beats_the_index_and_skips_illiquid():
    panel, nifty, truth = make_market()
    sim = fl.simulate(panel, nifty, oracle(truth), fl.Rules(min_traded_value=1e7))
    m = fl.metrics(sim['nav'].iloc[260:], nifty, sim['traded'])
    assert m['excess_cagr_pct'] > 10
    assert sim['invested_pct'] > 50


def test_illiquid_stock_never_bought(monkeypatch):
    panel, nifty, truth = make_market()
    bought = set()
    real = fl.fx.rank_to_score

    def spy(values):
        bought.update(values)
        return real(values)
    monkeypatch.setattr(fl.fx, 'rank_to_score', spy)
    fl.simulate(panel, nifty, oracle(truth), fl.Rules(min_traded_value=1e7))
    assert 59 not in bought and len(bought) > 0


def test_trend_gate_goes_to_cash_in_a_crash():
    panel, nifty, truth = make_market(crash_at=500)
    gated = fl.simulate(panel, nifty, oracle(truth), fl.Rules(min_traded_value=1e7))
    ungated = fl.simulate(panel, nifty, oracle(truth),
                          fl.Rules(min_traded_value=1e7, gate=False))
    assert gated['invested_pct'] < ungated['invested_pct']
    nav = gated['nav']
    flat = nav.iloc[540:560]
    assert (flat.max() - flat.min()) / flat.mean() < 1e-9   # sitting in cash


def test_costs_are_charged():
    panel, nifty, truth = make_market()
    cheap = fl.simulate(panel, nifty, oracle(truth), fl.Rules(min_traded_value=1e7, cost=0.0))
    dear = fl.simulate(panel, nifty, oracle(truth), fl.Rules(min_traded_value=1e7, cost=0.01))
    assert dear['nav'].iloc[-1] < cheap['nav'].iloc[-1]
    assert cheap['traded'] > 0


def test_fills_happen_at_next_open_not_the_signal_close():
    panel, nifty, truth = make_market()
    # one stock gaps up 30% at the open after every close: buying it at the
    # signal-day close would book that gap, buying at the next open must not
    panel['Open']['S58'] = panel['Close']['S58'].shift(1) * 1.3
    panel['Open'].iloc[0, 58] = 100.0
    only = lambda closes: {i: (1.0 if i == 58 else -1.0) for i in closes}
    rules = fl.Rules(min_traded_value=1e7, top_n=1, buffer=1)
    sim = fl.simulate(panel, nifty, only, rules)
    nav = sim['nav']
    first_buy = nav.iloc[261]
    assert first_buy < fl.START_CAPITAL * 1.0  # paid the gapped-up open + cost


def test_deflated_sharpe_separates_signal_from_noise():
    rng = np.random.default_rng(1)
    noise = rng.normal(0, 0.02, 120)
    signal = rng.normal(0.02, 0.02, 120)
    trials = [noise.mean() / noise.std(ddof=1), signal.mean() / signal.std(ddof=1),
              0.05, -0.05, 0.02]
    assert fl.deflated_sharpe(signal, trials) > 0.95
    assert fl.deflated_sharpe(noise, trials) < 0.6


def test_factor_definitions():
    c = np.linspace(100, 200, 300)                   # steady climb
    # same gain as one jump after many small down days: discrete, fades
    jumpy = np.r_[np.linspace(100, 95, 150), np.linspace(190, 185, 150)]
    assert fx.mom_12_1(c) == pytest.approx(c[-22] / c[-252] - 1)
    assert fx.discreteness(c) == pytest.approx(-1.0)
    assert fx.discreteness(jumpy) > 0.9
    assert fx.high_52w(c) == pytest.approx(1.0)
    assert fx.mom_12_1(c[:100]) is None
    s = fx.riskadj_scores({'a': c, 'b': c[::-1].copy(), 'c': c * 1.0001})
    assert s['a'] > s['b']


def test_run_and_report_end_to_end():
    panel, nifty, truth = make_market()
    variants = [('oracle', 'Oracle', oracle(truth), fl.Rules(min_traded_value=1e7)),
                ('mom', 'Momentum', fl.v_mom, fl.Rules(min_traded_value=1e7))]
    r = fl.run(panel, nifty, variants)
    assert set(r['variants']) == {'oracle', 'mom'}
    md = fl.to_markdown(r)
    assert 'Oracle' in md and 'survivorship' in md.lower()
    assert r['variants']['oracle']['prob_skill_dsr'] is not None


def test_sector_cap_limits_names_per_industry():
    panel, nifty, truth = make_market()
    sectors = {f"S{i:02d}": ('Banks' if i >= 40 else f"Other{i}") for i in range(N_STOCKS)}
    rules = fl.Rules(min_traded_value=1e7)
    sim = fl.simulate(panel, nifty, oracle(truth), rules, sectors)
    # the oracle wants the 15 best drifts, all of which are 'Banks' (40-58):
    # with the cap it must hold at most 4 of them, so it trails the uncapped run
    uncapped = fl.simulate(panel, nifty, oracle(truth), rules)
    assert sim['nav'].iloc[-1] < uncapped['nav'].iloc[-1]


def test_gate_on_index_steps_aside_in_a_crash():
    days = pd.bdate_range('2018-01-01', periods=900)
    r = np.full(900, 0.0008)
    r[500:600] = -0.01                       # a 100-day slide
    nifty = pd.Series(100 * np.cumprod(1 + r), index=days)
    g = fl.gated_index(nifty)
    assert g['gated_max_dd_pct'] > g['buyhold_max_dd_pct']    # smaller fall
    assert g['switches_per_year'] > 0
    calm = pd.Series(100 * np.cumprod(np.full(900, 1.0008)), index=days)
    g2 = fl.gated_index(calm)
    assert g2['gated_cagr_pct'] == pytest.approx(g2['buyhold_cagr_pct'], abs=0.2)


def test_half_exposure_sits_between_all_out_and_no_gate():
    panel, nifty, truth = make_market(crash_at=500)
    base = dict(min_traded_value=1e7)
    out = fl.simulate(panel, nifty, oracle(truth), fl.Rules(**base))
    half = fl.simulate(panel, nifty, oracle(truth), fl.Rules(low_exposure=0.5, **base))
    none = fl.simulate(panel, nifty, oracle(truth), fl.Rules(gate=False, **base))
    dd = lambda s: (s['nav'] / s['nav'].cummax() - 1).min()
    assert dd(out) >= dd(half) >= dd(none)          # falls get bigger
    assert out['invested_pct'] < half['invested_pct']
    crash = half['nav'].iloc[540:560]
    assert crash.max() - crash.min() > 0               # still half invested
    # back to full size after the trend recovers
    end = half['nav'].iloc[-60:]
    full = none['nav'].iloc[-60:]
    assert abs(end.pct_change().std() / full.pct_change().std() - 1) < 0.35


def test_half_gate_on_index():
    days = pd.bdate_range('2018-01-01', periods=900)
    r = np.full(900, 0.0008)
    r[500:600] = -0.01
    nifty = pd.Series(100 * np.cumprod(1 + r), index=days)
    out, half = fl.gated_index(nifty), fl.gated_index(nifty, low=0.5)
    assert out['gated_max_dd_pct'] >= half['gated_max_dd_pct'] >= half['buyhold_max_dd_pct']
