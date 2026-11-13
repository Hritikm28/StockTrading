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


def test_residual_momentum_ignores_the_market_move():
    import factors as fx
    rng = np.random.default_rng(3)
    m = np.cumprod(1 + rng.normal(0.001, 0.01, 520)) * 100
    noise = rng.normal(0, 0.005, 520)
    beta2 = np.cumprod(1 + 2 * np.diff(np.log(m), prepend=np.log(m[0])) + noise) * 50
    own = np.cumprod(1 + np.diff(np.log(m), prepend=np.log(m[0])) + noise + 0.002) * 50
    # a high-beta stock with no drift of its own scores far below a stock
    # that climbs on its own, whatever the market did
    assert fx.residual_mom(own, m) > fx.residual_mom(beta2, m) + 20
    assert fx.residual_mom(own[:-1], m) is None           # misaligned


def test_watch_list_stocks_are_never_bought():
    days = pd.bdate_range('2020-01-01', periods=400)
    rng = np.random.default_rng(0)
    cols = [f"S{i}" for i in range(20)]
    close = pd.DataFrame(np.cumprod(1 + rng.normal(0.001, 0.01, (400, 20)), axis=0)
                         * 100, index=days, columns=cols)
    watch = pd.DataFrame(0.0, index=days, columns=cols)
    watch['S0'] = 1.0
    panel = {'Close': close, 'Open': close, 'Value': close * 0 + 1e9,
             'Watch': watch}
    nifty = pd.Series(np.linspace(100, 200, 400), index=days)
    best_s0 = lambda closes: {i: (100.0 if i == 0 else float(i)) for i in closes}
    rules = fl.Rules(min_traded_value=1e7, top_n=3, buffer=3)
    seen = []

    def spy(closes):
        seen.append(set(closes))
        return best_s0(closes)

    fl.simulate(panel, nifty, spy, rules)
    assert any(0 in s for s in seen)                      # S0 is a candidate
    seen.clear()
    fl.simulate(panel, nifty, spy, fl.Rules(min_traded_value=1e7, top_n=3,
                                            buffer=3, skip_watch=True))
    assert seen and not any(0 in s for s in seen)         # never offered


def test_after_tax_charges_each_years_gain():
    days = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    nav = pd.Series(np.linspace(100, 172.8, len(days)), index=days)
    gross = ((nav.iloc[-1] / nav.iloc[0]) ** (365.25 / (days[-1] - days[0]).days) - 1) * 100
    assert fl.after_tax_cagr(nav) < gross
    flat = pd.Series(100.0, index=days)
    assert fl.after_tax_cagr(flat) == pytest.approx(0.0, abs=1e-9)
    assert fl.index_after_tax_cagr(nav) > fl.after_tax_cagr(nav)  # tax deferred


def test_review_every_ten_sessions_reviews_more_often_than_monthly():
    panel, nifty, truth = make_market()
    calls = {'m': 0, 'b': 0}

    def counting(key):
        def fn(closes):
            calls[key] += 1
            return {i: truth[i] for i in closes}
        return fn

    fl.simulate(panel, nifty, counting('m'), fl.Rules(min_traded_value=1e7))
    fl.simulate(panel, nifty, counting('b'),
                fl.Rules(min_traded_value=1e7, review_days=10))
    sessions = N_DAYS - 260
    assert calls['m'] == pytest.approx(sessions / 21, abs=3)     # ~monthly
    assert calls['b'] == pytest.approx(sessions / 10, abs=3)     # every 2 weeks


def test_more_holdings_are_actually_held():
    panel, nifty, truth = make_market()
    held = []

    def spy_rules(n):
        return fl.Rules(min_traded_value=1e7, top_n=n, buffer=2 * n,
                        universe_size=59)

    for n in (15, 25):
        sim = fl.simulate(panel, nifty, oracle(truth), spy_rules(n))
        # with n equal positions, the largest daily move shrinks as n grows
        held.append(sim['nav'].pct_change().abs().max())
    assert held[1] < held[0]


def test_steady_variant_prefers_calm_stocks_at_equal_strength():
    rng = np.random.default_rng(5)
    rm = rng.normal(0.0005, 0.008, 520)
    m = 100 * np.cumprod(1 + rm)
    calm = 100 * np.cumprod(1 + rm + 0.001 + rng.normal(0, 0.004, 520))
    wild = 100 * np.cumprod(1 + rm + 0.001 + rng.normal(0, 0.03, 520))
    filler = {i: 100 * np.cumprod(1 + rm + rng.normal(0, 0.01, 520))
              for i in range(2, 12)}
    closes = {0: calm, 1: wild, **filler}
    market = {k: m for k in closes}
    s = fl.v_resid_steady(closes, market)
    assert s[0] > s[1]


def test_risk_detail_metrics_are_reported():
    panel, nifty, truth = make_market()
    sim = fl.simulate(panel, nifty, oracle(truth), fl.Rules(min_traded_value=1e7))
    m = fl.metrics(sim['nav'].iloc[260:], nifty, sim['traded'])
    for k in ('sortino', 'calmar', 'longest_underwater_months', 'worst_12m_pct',
              'excess_first_half_pct', 'excess_second_half_pct'):
        assert m[k] is not None, k
    assert m['longest_underwater_months'] >= 0
