"""
Factor Lab — ten-year test of the live portfolio rules
======================================================
Live shadow grading needs months before it can say anything. This replays
the last ~10 years so a ranking idea can be judged in minutes, using the
same rules the live paper portfolio (portfolio_book.py) trades:

  - universe on each date: every NSE stock with a year of history whose
    60-day median traded value is at least Rs 5 crore, the 150 most traded
  - top 15 by the variant's score (positive half only), equal weight,
    replaced monthly with a top-30 buffer, filled at the next open
  - 200-DMA trend gate: buy only above it, cut back 2% below it
  - 0.22% cost per side (Indian delivery charges plus slippage, costs.py)

Every variant is scored with factors.py, the same code the live alphas use.

The stock data comes from NSE's daily bhav copies (bhav_history.py): every
stock that traded on each day, including those that later collapsed or
were delisted, so the test is free of survivorship bias. The old source,
today's NIFTY 500 from Yahoo, is still run for the live rules, to show how
much it flattered the results.

Honest limits, printed in the report:
  - Price returns only (no dividends) for both the strategy and NIFTY.
  - Sectors are today's NSE industry labels; past and delisted stocks
    without one are not capped.
  - After-tax figures are estimates: strategy gains taxed at 20% each year
    (it turns its holdings over about once a year), NIFTY at 12.5% once.
  - Each extra variant tried raises the bar for luck. The deflated Sharpe
    ratio below charges for every variant in this file, but not for ideas
    tried earlier and dropped, so it is still optimistic.

Writes research/factor_lab_report.md and research/factor_lab_report.json.

Usage (needs internet; runs in .github/workflows/factor_lab.yml):
    python factor_lab.py [--years 10]
"""

import argparse
import json
import math
from dataclasses import dataclass, asdict
from datetime import date, timedelta
from pathlib import Path
from statistics import NormalDist
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

import factors as fx

CACHE_DIR = Path("data/lab")
OUT_DIR = Path("research")
START_CAPITAL = 1_000_000.0
WINDOW = 520             # closes handed to each factor function (2 years)
STCG_RATE, INDEX_LTCG_RATE = 0.20, 0.125


@dataclass
class Rules:
    top_n: int = 15
    buffer: int = 30
    cost: float = 0.0022           # per side, see costs.py
    exit_band: float = 0.02
    gate: bool = True
    low_exposure: float = 0.0      # share kept invested while the trend is broken
    max_per_sector: int = 4
    universe_size: int = 150
    min_traded_value: float = 5e7  # Rs 5 crore
    liq_window: int = 60
    history: int = 260
    skip_watch: bool = False       # never buy NSE watch-list (BE/BZ) stocks


# ---------------------------------------------------------------------------
# Variants: closes dict -> raw score dict (higher = better)
# ---------------------------------------------------------------------------
def _apply(fn, closes):
    out = {}
    for k, c in closes.items():
        v = fn(c)
        if v is not None:
            out[k] = v
    return out


def v_mom(closes):
    return _apply(fx.mom_12_1, closes)


def v_riskadj(closes):
    return fx.riskadj_scores(closes)


def v_steady(closes):
    zm = fx.zscore(_apply(fx.mom_12_1, closes))
    zd = fx.zscore(_apply(fx.discreteness, closes))
    return {k: zm[k] - zd[k] for k in zm if k in zd}


def v_hi52(closes):
    return _apply(fx.high_52w, closes)


def v_riskadj_hi52(closes):
    za = fx.zscore(fx.riskadj_scores(closes))
    zh = fx.zscore(_apply(fx.high_52w, closes))
    return {k: (za[k] + zh[k]) / 2 for k in za if k in zh}


def v_resid(closes, market):
    out = {}
    for k, c in closes.items():
        v = fx.residual_mom(c, market.get(k))
        if v is not None:
            out[k] = v
    return out


v_resid.needs_market = True

LIVE = Rules(low_exposure=0.5)          # what portfolio_book trades today

VARIANTS = [
    ('live', 'Momentum 12-1, half invested when the trend breaks (live today)',
     v_mom, LIVE),
    ('live_nowatch', 'Live rules, never buying NSE watch-list stocks', v_mom,
     Rules(low_exposure=0.5, skip_watch=True)),
    ('resid', 'Residual momentum (less crash-prone), live rules', v_resid, LIVE),
    ('resid_nowatch', 'Residual momentum, never buying watch-list stocks',
     v_resid, Rules(low_exposure=0.5, skip_watch=True)),
    ('live_nogate', 'Momentum 12-1 without the trend gate', v_mom,
     Rules(gate=False)),
]


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate(panel: Dict[str, pd.DataFrame], nifty: pd.Series,
             score_fn: Callable, rules: Rules,
             sectors: Optional[Dict[str, str]] = None) -> Dict:
    """Day-by-day replay of the portfolio rules. Decisions use closes up to
    day t and fill at day t+1's open.

    Exposure is 100% while NIFTY is above its 200-day average. When it falls
    exit_band below, every holding is cut to low_exposure (0 = sell all, the
    live rule; 0.5 = sell half) and new names are bought at that size; when
    NIFTY climbs back above, holdings are scaled back up."""
    days = panel['Close'].index.intersection(nifty.index)
    close = panel['Close'].reindex(days)
    opn = panel['Open'].reindex(days)
    value = (panel['Value'].reindex(days) if 'Value' in panel
             else close * panel['Volume'].reindex(days))
    tv = value.rolling(rules.liq_window,
                       min_periods=int(rules.liq_window * 0.7)).median()
    nifty = nifty.reindex(days).ffill()
    vs200 = (nifty / nifty.rolling(200).mean() - 1).to_numpy()
    cols = list(close.columns)
    sector_of = [(sectors or {}).get(c) for c in cols]
    C, O, TV = (close.to_numpy(dtype=float), opn.to_numpy(dtype=float),
                tv.to_numpy(dtype=float))
    W = (panel['Watch'].reindex(days).to_numpy(dtype=float) == 1
         if 'Watch' in panel else np.zeros(C.shape, dtype=bool))
    N = nifty.to_numpy(dtype=float)
    needs_market = getattr(score_fn, 'needs_market', False)
    valid_count = np.cumsum(np.isfinite(C), axis=0)

    cash, holdings, pending = START_CAPITAL, {}, []
    exposure: Optional[float] = None
    last_px: Dict[int, float] = {}
    last_month, nav, traded, invested_days = None, [], 0.0, 0
    cost = rules.cost

    def pick(t) -> List[int]:
        ok = (valid_count[t] >= rules.history) & (TV[t] >= rules.min_traded_value)
        cand = np.where(ok)[0]
        cand = cand[np.argsort(-TV[t, cand])][:rules.universe_size]
        if rules.skip_watch:              # dropped after the universe is set
            cand = cand[~W[t, cand]]
        closes, market = {}, {}
        lo = max(0, t - WINDOW + 1)
        for i in cand:
            w = C[lo:t + 1, i]
            keep = np.isfinite(w)
            if needs_market:
                keep &= np.isfinite(N[lo:t + 1])
                market[i] = N[lo:t + 1][keep]
            closes[i] = w[keep]
        raw = score_fn(closes, market) if needs_market else score_fn(closes)
        scores = fx.rank_to_score(raw)
        ranked = sorted((i for i, s in scores.items() if s > 0),
                        key=lambda i: -scores[i])
        return ranked

    for t in range(len(days)):
        # 1. fill yesterday's decisions at today's open (sells first)
        # S = sell all, P = sell a fraction, B = buy, T = top up a holding
        if pending:
            left = []
            for order in sorted(pending, key=lambda p: p[0] not in 'SP'):
                side, i, value = order
                px = O[t, i] if np.isfinite(O[t, i]) else C[t, i]
                if not np.isfinite(px) or px <= 0:
                    if side in 'SP' and i in holdings:
                        left.append(order)  # no trade today: sells retry
                    continue                # buys lapse
                if side in 'SP' and i in holdings:
                    n = holdings[i] if side == 'S' else holdings[i] * value
                    cash += n * px * (1 - cost)
                    traded += n * px
                    holdings[i] -= n
                    if side == 'S' or value >= 1:
                        holdings.pop(i)
                elif side in 'BT' and (i in holdings) == (side == 'T'):
                    v = min(value, cash / (1 + cost))
                    if v > 0:
                        holdings[i] = holdings.get(i, 0.0) + v / px
                        last_px.setdefault(i, px)
                        cash -= v * (1 + cost)
                        traded += v
            pending = left
        # 2. mark at today's close
        for i in holdings:
            if np.isfinite(C[t, i]):
                last_px[i] = C[t, i]
        value = cash + sum(n * last_px[i] for i, n in holdings.items())
        nav.append(value)
        if t >= rules.history:
            invested_days += bool(holdings)
        if t < rules.history or not np.isfinite(vs200[t]) or t == len(days) - 1:
            continue
        # 3. decide for tomorrow's open
        month = (days[t].year, days[t].month)
        v = vs200[t]
        prev = exposure
        if not rules.gate:
            exposure = 1.0
        elif prev is None:
            exposure = 1.0 if v > 0 else rules.low_exposure
        elif prev == 1.0 and v < -rules.exit_band:
            exposure = rules.low_exposure
        elif prev < 1.0 and v > 0:
            exposure = 1.0
        if prev is not None and exposure != prev and holdings:
            if exposure < prev:              # trend broke: cut every holding
                pending = [('P', i, 1 - exposure / prev) for i in holdings]
            elif prev > 0:                   # trend back: scale holdings up
                pending = [('T', i, holdings[i] * last_px[i] * (exposure / prev - 1))
                           for i in holdings]
            continue
        can_buy = exposure > 0 and (exposure < 1.0 or not rules.gate or v > 0)
        rebalance = bool(holdings) and month != last_month
        entering = not holdings and can_buy
        if not (rebalance or entering):
            continue
        ranked = pick(t)
        rank_of = {i: r + 1 for r, i in enumerate(ranked)}
        keep = [i for i in holdings if rank_of.get(i, 10 ** 9) <= rules.buffer]
        sells = [i for i in holdings if i not in keep]
        buys = []
        if can_buy:
            per_sector: Dict[str, int] = {}
            for i in keep:
                if sector_of[i]:
                    per_sector[sector_of[i]] = per_sector.get(sector_of[i], 0) + 1
            for i in ranked:
                if len(keep) + len(buys) >= rules.top_n:
                    break
                sec = sector_of[i]
                if i in keep or (sec and per_sector.get(sec, 0) >= rules.max_per_sector):
                    continue
                buys.append(i)
                if sec:
                    per_sector[sec] = per_sector.get(sec, 0) + 1
        freed = sum(holdings[i] * last_px[i] for i in sells) * (1 - cost)
        per = 0.0
        if buys:
            per = min(exposure * value / rules.top_n,
                      (cash + freed) / (len(buys) * (1 + cost)))
        pending = [('S', i, 0.0) for i in sells] + [('B', i, per) for i in buys]
        last_month = month

    nav = pd.Series(nav, index=days)
    return {'nav': nav, 'traded': traded,
            'invested_pct': invested_days / max(len(days) - rules.history, 1) * 100}


def gated_index(nifty: pd.Series, band: float = 0.02, cost: float = 0.001,
                low: float = 0.0) -> Dict:
    """The trend gate applied to NIFTY itself: fully invested while the
    index is above its 200-day average, cut to `low` (0 = all out) when it
    falls `band` below, back to full above. Uses no stock list, so it
    carries no survivorship bias and isolates what the gate alone adds or
    costs."""
    n = nifty.dropna()
    vs = (n / n.rolling(200).mean() - 1).to_numpy()
    r = n.pct_change().fillna(0).to_numpy()
    nav, pos, switches = [1.0], low, 0
    for t in range(1, len(n)):
        v = nav[-1] * (1 + pos * r[t])
        want = pos
        if np.isfinite(vs[t]):
            if pos == 1.0 and vs[t] < -band:
                want = low
            elif pos < 1.0 and vs[t] > 0:
                want = 1.0
        if want != pos:                 # decided at today's close
            v *= 1 - cost * abs(want - pos)
            switches += 1
            pos = want
        nav.append(v)
    s = pd.Series(nav, index=n.index).iloc[200:]
    bh = n.iloc[200:]
    years = (s.index[-1] - s.index[0]).days / 365.25
    cagr = lambda x: ((x.iloc[-1] / x.iloc[0]) ** (1 / years) - 1) * 100
    mdd = lambda x: (x / x.cummax() - 1).min() * 100
    return {'gated_cagr_pct': round(cagr(s), 2), 'buyhold_cagr_pct': round(cagr(bh), 2),
            'gated_max_dd_pct': round(mdd(s), 1), 'buyhold_max_dd_pct': round(mdd(bh), 1),
            'switches_per_year': round(switches / years, 1)}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def _moments(x: np.ndarray):
    m, s = x.mean(), x.std(ddof=0)
    if s == 0:
        return 0.0, 3.0
    return float(((x - m) ** 3).mean() / s ** 3), float(((x - m) ** 4).mean() / s ** 4)


def deflated_sharpe(x: np.ndarray, trial_srs: List[float]) -> Optional[float]:
    """Bailey & Lopez de Prado (2014): probability that the true Sharpe of
    the series x is above what the best of len(trial_srs) skill-less
    variants would show by luck. x and trial_srs share a frequency."""
    x = np.asarray(x, dtype=float)
    if len(x) < 12 or x.std(ddof=1) == 0:
        return None
    nd = NormalDist()
    sr = x.mean() / x.std(ddof=1)
    skew, kurt = _moments(x)
    n = len(trial_srs)
    sr0 = 0.0
    if n > 1:
        g = 0.5772156649
        sd = float(np.std(trial_srs, ddof=1))
        sr0 = sd * ((1 - g) * nd.inv_cdf(1 - 1 / n)
                    + g * nd.inv_cdf(1 - 1 / (n * math.e)))
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr ** 2
    if denom <= 0:
        return None
    return float(nd.cdf((sr - sr0) * math.sqrt(len(x) - 1) / math.sqrt(denom)))


def after_tax_cagr(nav: pd.Series, rate: float = STCG_RATE) -> float:
    """CAGR after tax when each calendar year's gain is taxed at `rate`
    (losses carried forward). Assumes gains are realised every year, which
    is close to true for a book that is turned over about once a year."""
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    ends = nav.groupby(nav.index.year).last()
    value, prev, carry = 1.0, nav.iloc[0], 0.0
    for end in ends:
        gain = value * (end / prev - 1)
        value += gain
        taxable = gain + carry
        if taxable > 0:
            value -= rate * taxable
            carry = 0.0
        else:
            carry = taxable
        prev = end
    return (value ** (1 / years) - 1) * 100


def index_after_tax_cagr(series: pd.Series, rate: float = INDEX_LTCG_RATE) -> float:
    """Buy and hold, taxed once on the whole gain when sold at the end."""
    years = (series.index[-1] - series.index[0]).days / 365.25
    g = series.iloc[-1] / series.iloc[0]
    final = g - rate * max(g - 1, 0)
    return (final ** (1 / years) - 1) * 100


def metrics(nav: pd.Series, nifty: pd.Series, traded: float) -> Dict:
    nav = nav.dropna()
    nifty = nifty.reindex(nav.index).ffill()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1
    n_cagr = (nifty.iloc[-1] / nifty.iloc[0]) ** (1 / years) - 1
    d = nav.pct_change().dropna()
    dd = (nav / nav.cummax() - 1).min()
    n_dd = (nifty / nifty.cummax() - 1).min()
    def period_returns(s, freq):
        ends = s.groupby(s.index.to_period(freq)).last()
        prev = ends.shift(1)
        prev.iloc[0] = s.iloc[0]
        return ends / prev - 1

    m, nm = period_returns(nav, 'M'), period_returns(nifty, 'M')
    exc = (m - nm).dropna()
    t_exc = (exc.mean() / (exc.std(ddof=1) / math.sqrt(len(exc)))
             if len(exc) > 2 and exc.std(ddof=1) > 0 else None)
    y, ny = period_returns(nav, 'Y'), period_returns(nifty, 'Y')
    yearly = {int(p.year): [round(y[p] * 100, 1), round(ny[p] * 100, 1)]
              for p in y.index}
    return {
        'cagr_pct': round(cagr * 100, 2), 'nifty_cagr_pct': round(n_cagr * 100, 2),
        'excess_cagr_pct': round((cagr - n_cagr) * 100, 2),
        'after_tax_cagr_pct': round(after_tax_cagr(nav), 2),
        'nifty_after_tax_cagr_pct': round(index_after_tax_cagr(nifty), 2),
        'vol_pct': round(d.std() * math.sqrt(252) * 100, 1),
        'sharpe': round(d.mean() / d.std() * math.sqrt(252), 2) if d.std() else None,
        'max_dd_pct': round(dd * 100, 1), 'nifty_max_dd_pct': round(n_dd * 100, 1),
        'turnover_pct_yr': round(traded / nav.mean() / years * 100 / 2, 0),
        'beat_nifty_months_pct': round((exc > 0).mean() * 100, 1),
        't_excess': round(t_exc, 2) if t_exc is not None else None,
        'yearly': yearly, '_monthly_excess': exc.to_numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# Data (runs in the cloud)
# ---------------------------------------------------------------------------
def load_nifty(years: int) -> pd.Series:
    import yfinance as yf
    start = (date.today() - timedelta(days=int(365.25 * years) + 400)).isoformat()
    n = yf.download('^NSEI', start=start, auto_adjust=False, progress=False)
    nifty = n['Close']
    if isinstance(nifty, pd.DataFrame):
        nifty = nifty.iloc[:, 0]
    nifty.index = pd.to_datetime(nifty.index).tz_localize(None)
    return nifty


def load_bias_free_panel(years: int, min_value: float = Rules.min_traded_value
                         ) -> tuple:
    """Every NSE stock from the bhav copies (bhav_history.py), adjusted for
    splits and bonuses, with traded value and the watch-list flag. Only
    stocks that ever traded Rs `min_value` in a day are kept."""
    import bhav_history as bh
    from market_calendar import latest_session
    from universe import fetch_nifty500_list

    start = date.today() - timedelta(days=int(365.25 * years) + 400)
    bh.update(start, latest_session())
    rows = bh.load(start)
    rows = rows[rows['date'] >= pd.Timestamp(start)]
    panel = bh.build_panel(rows, min_value=min_value)
    lst = fetch_nifty500_list()
    sectors = ({} if lst is None or lst.empty else
               dict(zip(lst['Symbol'].astype(str), lst.get('Industry', ''))))
    return panel, load_nifty(years), sectors


def load_panel(years: int) -> tuple:
    """10-year Open/Close/Volume for TODAY's NIFTY 500 from Yahoo (biased:
    no stock that collapsed and left the index), NIFTY, and each stock's
    NSE industry. Cached for a week."""
    import yfinance as yf
    from universe import fetch_nifty500_list

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = CACHE_DIR / f"panel_{years}y.pkl"
    if cache.exists() and (date.today() - date.fromtimestamp(
            cache.stat().st_mtime)).days < 7:
        return pd.read_pickle(cache)

    lst = fetch_nifty500_list(force=True)
    if lst is None or lst.empty:
        raise SystemExit("Could not download the NIFTY 500 list from NSE")
    sectors = dict(zip(lst['Symbol'].astype(str), lst.get('Industry', '')))
    symbols = sorted(f"{s}.NS" for s in lst['Symbol'].dropna().astype(str))
    start = (date.today() - timedelta(days=int(365.25 * years) + 400)).isoformat()
    parts = {'Open': [], 'Close': [], 'Volume': []}
    for k in range(0, len(symbols), 50):
        chunk = symbols[k:k + 50]
        df = yf.download(chunk, start=start, auto_adjust=False,
                         progress=False, threads=True)
        if df is None or df.empty:
            continue
        for f in parts:
            sub = df[f]
            if isinstance(sub, pd.Series):
                sub = sub.to_frame(chunk[0])
            parts[f].append(sub)
        print(f"  downloaded {min(k + 50, len(symbols))}/{len(symbols)}")
    panel = {f: pd.concat(v, axis=1).sort_index() for f, v in parts.items()}
    for f in panel:
        panel[f].columns = [str(c).replace('.NS', '') for c in panel[f].columns]
        panel[f].index = pd.to_datetime(panel[f].index).tz_localize(None)
    n = yf.download('^NSEI', start=start, auto_adjust=False, progress=False)
    nifty = n['Close']
    if isinstance(nifty, pd.DataFrame):
        nifty = nifty.iloc[:, 0]
    nifty.index = pd.to_datetime(nifty.index).tz_localize(None)
    pd.to_pickle((panel, nifty, sectors), cache)
    return panel, nifty, sectors


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def run(panel, nifty, variants=VARIANTS, sectors=None, biased=None) -> Dict:
    """`biased`: optional (panel, nifty, sectors) from today's NIFTY 500;
    the live rules are replayed on it to measure survivorship bias."""
    results = {}
    for key, label, fn, rules in variants:
        sim = simulate(panel, nifty, fn, rules, sectors)
        nav = sim['nav'].iloc[rules.history:]
        m = metrics(nav, nifty, sim['traded'])
        m.update(label=label, invested_pct=round(sim['invested_pct'], 1),
                 rules=asdict(rules))
        results[key] = m
        print(f"  {key:<18} CAGR {m['cagr_pct']:6.2f}% vs NIFTY "
              f"{m['nifty_cagr_pct']:6.2f}%  maxDD {m['max_dd_pct']}%")
    monthly_srs = []
    for m in results.values():
        x = np.array(m['_monthly_excess'])
        if len(x) > 2 and x.std(ddof=1) > 0:
            monthly_srs.append(x.mean() / x.std(ddof=1))
    for m in results.values():
        x = np.array(m.pop('_monthly_excess'))
        m['prob_skill_dsr'] = deflated_sharpe(x, monthly_srs)
        m['prob_skill_dsr'] = (None if m['prob_skill_dsr'] is None
                               else round(m['prob_skill_dsr'] * 100, 1))
    days = panel['Close'].index.intersection(nifty.index)
    first = days[min(Rules().history, len(days) - 1)]
    bias = None
    if biased is not None:
        bp, bn, bs = biased
        sim = simulate(bp, bn, v_mom, LIVE, bs)
        nav = sim['nav'].iloc[LIVE.history:]
        nav = nav[nav.index >= first]
        if len(nav) > 260:
            m = metrics(nav, bn, sim['traded'])
            bias = {'cagr_pct': m['cagr_pct'], 'max_dd_pct': m['max_dd_pct'],
                    'n_candidates': int(bp['Close'].shape[1])}
            print(f"  live rules on today's NIFTY 500 (biased): CAGR {m['cagr_pct']}%")
    return {'generated': date.today().isoformat(),
            'period': [str(first.date()), str(days[-1].date())],
            'n_candidates': int(panel['Close'].shape[1]),
            'n_variants': len(results), 'variants': results,
            'survivorship': bias,
            'gate_on_nifty': {
                'all_out': gated_index(nifty.reindex(days).ffill()),
                'half': gated_index(nifty.reindex(days).ffill(), low=0.5)}}


def to_markdown(r: Dict) -> str:
    v = r['variants']
    lines = ["# Factor Lab: ten-year test of the portfolio rules", "",
             f"_Generated {r['generated']} by factor_lab.py. Period "
             f"{r['period'][0]} to {r['period'][1]}, {r['n_candidates']} "
             f"stocks from NSE's daily bhav copies, including those that "
             f"later collapsed or were delisted (no survivorship bias). Same "
             f"rules as the live paper portfolio: top 15 (max 4 per "
             f"industry), monthly replacement with a top-30 buffer, trend "
             f"gate, 0.22% cost per side._", "",
             "Price returns only, for the strategy and NIFTY alike. "
             "After-tax figures are estimates: strategy gains taxed at 20% "
             "each year, NIFTY held throughout and taxed 12.5% once at the "
             "end.", "",
             "| Variant | CAGR | After tax | NIFTY | NIFTY after tax | Excess "
             "| Max fall | Sharpe | Turnover/yr | Months beating NIFTY | "
             "Chance it's skill |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, m in sorted(v.items(), key=lambda kv: -kv[1]['excess_cagr_pct']):
        p = m['prob_skill_dsr']
        lines.append(
            f"| {m['label']} | {m['cagr_pct']}% | {m.get('after_tax_cagr_pct')}% | "
            f"{m['nifty_cagr_pct']}% | {m.get('nifty_after_tax_cagr_pct')}% | "
            f"**{m['excess_cagr_pct']:+}%** | {m['max_dd_pct']}% | "
            f"{m['sharpe']} | {m['turnover_pct_yr']:.0f}% | "
            f"{m['beat_nifty_months_pct']}% | "
            f"{'—' if p is None else f'{p}%'} |")
    sb = r.get('survivorship')
    live = v.get('live')
    if sb and live:
        lines += ["", f"**Survivorship bias, measured:** the same live rules "
                  f"tested the old way, on today's NIFTY 500 only "
                  f"({sb['n_candidates']} stocks), returned {sb['cagr_pct']}% "
                  f"a year (worst fall {sb['max_dd_pct']}%), against "
                  f"{live['cagr_pct']}% here. The old test overstated the "
                  f"yearly return by {sb['cagr_pct'] - live['cagr_pct']:.1f} "
                  f"points."]
    lines += ["", "_\"Chance it's skill\" is the deflated Sharpe ratio of the "
              "monthly gap to NIFTY: the probability the edge is real after "
              "charging for every variant in this table. 95%+ is "
              "convincing; under 50% means luck explains it as well as "
              "skill. It does not charge for ideas tried before this lab "
              "existed, so read it as an upper bound too._", "",
              "## Year by year (strategy % / NIFTY %)", ""]
    g = r.get('gate_on_nifty')
    years = sorted({y for m in v.values() for y in m['yearly']})
    lines.append("| Variant | " + " | ".join(str(y) for y in years) + " |")
    lines.append("|---|" + "---|" * len(years))
    for k, m in v.items():
        cells = []
        for y in years:
            s = m['yearly'].get(y) or m['yearly'].get(str(y))
            cells.append('' if not s else f"{s[0]} / {s[1]}")
        lines.append(f"| {m['label']} | " + " | ".join(cells) + " |")
    if g:
        out, half = g['all_out'], g['half']
        lines += ["", "## The trend gate on NIFTY itself (no survivorship bias)", "",
                  "Holding NIFTY fully while it is above its 200-day average "
                  "and cutting back when it falls 2% below (0.1% cost per "
                  "switch), versus simply holding it. No stock list is "
                  "involved, so this isolates what the gate adds or costs.", "",
                  "| | Yearly return | Worst fall |", "|---|---|---|",
                  f"| Gate: sell everything | {out['gated_cagr_pct']}% | {out['gated_max_dd_pct']}% |",
                  f"| Gate: sell half | {half['gated_cagr_pct']}% | {half['gated_max_dd_pct']}% |",
                  f"| NIFTY held throughout | {out['buyhold_cagr_pct']}% | {out['buyhold_max_dd_pct']}% |",
                  "", f"The gate switched {out['switches_per_year']} times a year."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--years', type=int, default=10)
    ap.add_argument('--skip-biased', action='store_true',
                    help="don't replay the live rules on today's NIFTY 500")
    args = ap.parse_args(argv)
    ap_biased = None
    print(f"Loading {args.years}-year bias-free panel from NSE bhav copies...")
    panel, nifty, sectors = load_bias_free_panel(args.years)
    print(f"Panel: {panel['Close'].shape[1]} stocks, "
          f"{len(panel['Close'])} days. Simulating {len(VARIANTS)} variants...")
    if not args.skip_biased:
        try:
            print("Loading today's NIFTY 500 from Yahoo (to measure the bias)...")
            ap_biased = load_panel(args.years)
        except Exception as e:
            print(f"  skipped: {e}")
    report = run(panel, nifty, sectors=sectors, biased=ap_biased)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "factor_lab_report.json").write_text(
        json.dumps(report, indent=1, default=str))
    (OUT_DIR / "factor_lab_report.md").write_text(to_markdown(report))
    print(f"Wrote {OUT_DIR}/factor_lab_report.md")


if __name__ == '__main__':
    main()
