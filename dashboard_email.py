"""
Turns paper_trading/results/dashboard.json (market_dashboard.py) and the
signal grades into the email sections and the dashboard chart.

Everything is written for a reader who is not a trader: each number comes
with what it means. Text that came from outside (company filings) is
escaped before it goes into the email.
"""

import html
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

GREEN, AMBER, RED, NAVY, GREY, ORANGE = (
    '#1a7a3a', '#b8860b', '#b0322b', '#1a2b4a', '#888888', '#d08a00')
COLOR = {'green': GREEN, 'amber': AMBER, 'red': RED}
CHART_FILE = Path('market_dashboard.png')

PROMOTE_MIN_IC, PROMOTE_MIN_T, PROMOTE_MIN_DAYS = 0.03, 2.0, 120
SIGNAL_NAMES = {
    'resid_mom': 'Residual momentum (rising on its own strength)',
    'momentum': 'Plain momentum (price rose most)',
    'mom_riskadj': 'Steady momentum (rise divided by bumpiness)',
    'quality': 'Quality (profitable, low debt)',
    'pead': 'Earnings surprise (results beat forecasts)',
    'insider': 'Insider buying (owners and directors buying)',
    'pledge': 'Promoter pledging (owners borrowing against shares)',
    'delivery_pct': 'Delivery buying (shares kept, not day-traded)',
    'hi52': 'Near its 52-week high',
    'mean_rev': 'Bounce-back (fell too far, too fast)',
    'sector_mom': 'Sector momentum (its whole sector is rising)',
}


def _pct(v, nd=1, points=False) -> str:
    if v is None:
        return '—'
    color = GREEN if v > 0 else RED if v < 0 else '#555'
    unit = ' pts' if points else '%'
    return f"<span style='color:{color}'>{v:+.{nd}f}{unit}</span>"


def _plain(v, nd=1, points=False) -> str:
    """A signed change without colour, for things where up isn't 'good'
    (oil, the dollar, interest rates, fear gauges)."""
    if v is None:
        return '—'
    return f"{v:+.{nd}f}{' pts' if points else '%'}"


def _table(headers, rows) -> str:
    head = "".join(f"<th align='{a}'>{h}</th>" for h, a in headers)
    body = ""
    for i, cells in enumerate(rows):
        bg = '#f4f7fb' if i % 2 else '#ffffff'
        body += (f"<tr style='background:{bg}'>"
                 + "".join(f"<td align='{a}'>{c}</td>"
                           for c, (_, a) in zip(cells, headers)) + "</tr>")
    return (f"<table cellpadding='6' cellspacing='0' "
            f"style='border-collapse:collapse;width:100%;font-size:14px'>"
            f"<tr style='background:{NAVY};color:#fff'>{head}</tr>{body}</table>")


def _h(title: str) -> str:
    return f"<h3 style='margin-bottom:4px'>{title}</h3>"


def _note(text: str) -> str:
    return f"<p style='font-size:12px;color:#555;margin:4px 0'>{text}</p>"


# ---------------------------------------------------------------------------
# Market health
# ---------------------------------------------------------------------------
def lights_summary(d: Dict) -> str:
    """One line for the daily email."""
    lights = d.get('lights') or []
    if not lights:
        return ""
    n = {s: sum(1 for l in lights if l['status'] == s) for s in COLOR}
    dots = "".join(f"<span style='color:{COLOR[l['status']]}' "
                   f"title='{html.escape(l['name'])}'>●</span>" for l in lights)
    worst = [l for l in lights if l['status'] == 'red']
    extra = (" Red: " + ", ".join(l['name'].lower() for l in worst) + "."
             if worst else "")
    return (f"<p style='font-size:13px;margin:4px 0'>Market health {dots} "
            f"{n['green']} green, {n['amber']} amber, {n['red']} red.{extra} "
            "<span style='color:#888'>(Context only; details in Sunday's "
            "email.)</span></p>")


def lights_html(d: Dict) -> str:
    lights = d.get('lights') or []
    if not lights:
        return ""
    rows = [(f"<span style='color:{COLOR[l['status']]};font-size:18px'>●</span>",
             f"<b>{html.escape(l['name'])}</b>", html.escape(l['text']))
            for l in lights]
    return (_h("Market health check")
            + _table([('', 'center'), ('Gauge', 'left'), ('What it says', 'left')], rows)
            + _note("Green is healthy, amber is mixed, red is a warning. These "
                    "are the gauges big funds watch, as context: the system "
                    "does not trade on them, because they haven't passed our "
                    "ten-year test."))


def market_detail_html(d: Dict) -> str:
    out: List[str] = []
    b = d.get('breadth')
    if b and b.get('pct_above_200dma') is not None and b.get('pct_above_50dma') is not None:
        out.append(
            f"<b>Breadth:</b> {b['pct_above_200dma']:.0f}% of {b['n_stocks']} stocks "
            "are above their 200-day average"
            + (f" ({b['pct_above_200dma_month_ago']:.0f}% a month ago)"
               if b.get('pct_above_200dma_month_ago') is not None else "")
            + f"; {b['pct_above_50dma']:.0f}% above their 50-day average. "
            f"{b['new_highs']} hit a one-year high, {b['new_lows']} a one-year low.")
    v = (d.get('valuation') or {})
    n = v.get('NIFTY 50')
    if n and all(n.get(k) is not None for k in ('pe', 'pe_median', 'pe_low', 'pe_high')):
        line = (f"<b>Price tag:</b> NIFTY trades at {n['pe']:.1f} times profits "
                f"(usual since {n['since'][:4]}: {n['pe_median']:.1f}; range "
                f"{n['pe_low']:.0f} to {n['pe_high']:.0f})")
        if n.get('div_yield') is not None:
            line += f", pays {n['div_yield']:.1f}% in dividends"
        for label, word in (('Midcap 150', 'mid-sized'), ('Smallcap 250', 'small')):
            m = v.get(label)
            if m and m.get('pe') is not None:
                line += (f"; {word} companies {m['pe']:.1f} times"
                         + (f" (pricier than {m['pe_percentile']:.0f}% of days)"
                            if m.get('pe_percentile') is not None else ""))
        out.append(line + ".")
    f = d.get('flows')
    if f and all(f.get(k) is not None for k in ('fii_5d', 'fii_20d', 'dii_5d', 'dii_20d')):
        def money(x):
            return f"{'+' if x >= 0 else '−'}₹{abs(x):,.0f} crore"
        out.append(f"<b>Big money:</b> foreign investors {money(f['fii_5d'])} this "
                   f"week and {money(f['fii_20d'])} over 20 days; Indian funds "
                   f"{money(f['dii_5d'])} and {money(f['dii_20d'])}. "
                   "(Plus means net buying.)")
    p = d.get('positioning')
    if p and p.get('pcr') is not None:
        line = (f"<b>Options market:</b> put-call ratio {p['pcr']:.2f} "
                f"(20-day average {p.get('pcr_avg_20d', p['pcr']):.2f}). Above 1 "
                "means more protection is being bought or sold than bets on a rise")
        if p.get('retail_long_pct') is not None:
            line += (f". Small traders have {p['retail_long_pct']:.0f}% of their "
                     "index-futures bets on a rise")
        out.append(line + ".")
    g = (d.get('global') or {}).get('rows')
    html_out = ""
    if out:
        html_out += ("<ul style='font-size:14px;line-height:1.6'>"
                     + "".join(f"<li>{t}</li>" for t in out) + "</ul>")
    if g:
        rows = [(html.escape(r['name']), f"{r['last']:,.2f}",
                 _plain(r['week'], 2 if r['points'] else 1, r['points']),
                 _plain(r['month'], 2 if r['points'] else 1, r['points'])) for r in g]
        html_out += (_h("Around the world")
                     + _table([('', 'left'), ('Now', 'right'), ('1 week', 'right'),
                               ('1 month', 'right')], rows)
                     + _note("Oil and the dollar matter to India: dearer oil and a "
                             "weaker rupee usually hurt; falling US rates usually help."))
    return html_out


def sectors_html(d: Dict) -> str:
    rows = d.get('sectors') or []
    if not rows:
        return ""

    def tab(items):
        return _table([('', 'left'), ('1 week', 'right'), ('1 month', 'right'),
                       ('3 months', 'right')],
                      [(html.escape(r['name']), _pct(r['week']), _pct(r['month']),
                        _pct(r['quarter'])) for r in items])

    sec = [r for r in rows if r['group'] == 'sector']
    style = [r for r in rows if r['group'] == 'style']
    size = [r for r in rows if r['group'] == 'size']
    out = ""
    if sec:
        shown = sec if len(sec) <= 8 else sec[:4] + sec[-4:]
        out += (_h("Sectors: this month's best and worst") + tab(shown))
    if style:
        out += (_h("Which style of investing is working")
                + tab(style)
                + _note("Each row is an NSE index that follows one style. "
                        "'Momentum' is the style our system uses; when it is near "
                        "the bottom, expect our picks to lag too."))
    if size:
        out += _h("Big companies vs small") + tab(size)
    return out


# ---------------------------------------------------------------------------
# Portfolio X-ray
# ---------------------------------------------------------------------------
def risk_html(d: Dict) -> str:
    r = d.get('risk') or {}
    need = ('beta', 'vol_pct', 'nifty_vol_pct', 'var95_rupees', 'var95_pct',
            'worst_day_pct', 'worst_day', 'top_weight_pct')
    if any(r.get(k) is None for k in need):
        return ""

    def rs(x):
        return f"₹{abs(x):,.0f}"

    items = [
        f"<b>How jumpy:</b> it moves about {r['beta']:.1f} times as much as NIFTY. "
        f"Its yearly swing is {r['vol_pct']:.0f}% against NIFTY's "
        f"{r['nifty_vol_pct']:.0f}%.",
        f"<b>A bad day:</b> about one day in twenty, expect to lose "
        f"{rs(r['var95_rupees'])} or more ({r['var95_pct']:.1f}%)"
        + (f"; on those days the average loss was {r['bad_day_avg_pct']:.1f}%"
           if r.get('bad_day_avg_pct') is not None else "")
        + f". The worst single day for these stocks in the past year was "
          f"{r['worst_day_pct']:.1f}% ({r['worst_day']}).",
    ]
    if r.get('if_nifty_falls_10_pct') is not None:
        line = (f"<b>Stress test:</b> if NIFTY fell 10%, expect roughly "
                f"{r['if_nifty_falls_10_pct']:.0f}%.")
        w = r.get('replay_worst_week')
        if w:
            line += (f" Replay: in NIFTY's worst week of the past year "
                     f"({w['from']} to {w['to']}, {w['nifty_pct']:.1f}%), these "
                     f"stocks would have moved {w['portfolio_pct']:+.1f}%.")
        items.append(line)
    if r.get('avg_correlation') is not None:
        c = r['avg_correlation']
        items.append(
            f"<b>Are the eggs in one basket?</b> Our stocks move together "
            f"{c:.2f} on a scale from 0 (independently) to 1 (as one), which is "
            + ("well spread." if c < 0.25 else "reasonably spread." if c < 0.45
               else "quite alike, so they can all fall together.")
            + f" Biggest holding: {r['top_weight_pct']:.0f}% of the money."
            + (" Biggest industries: " + ", ".join(
                f"{html.escape(s['name'])} {s['pct']:.0f}%" for s in r['sectors'][:3]) + "."
               if r.get('sectors') else ""))
    if r.get('volatility_percentile') is not None:
        items.append(
            f"<b>What kind of stocks:</b> a typical holding is bumpier than "
            f"{r['volatility_percentile']:.0f}% of NIFTY 500 stocks"
            + (f" and has risen more over the past year than "
               f"{r['momentum_percentile']:.0f}% of them."
               if r.get('momentum_percentile') is not None else "."))
    c = r.get('contributors') or []
    if c:
        def names(rows):
            return ", ".join(f"{html.escape(x['symbol'])} "
                             f"({'+' if x['rupees'] >= 0 else '−'}{rs(x['rupees'])})"
                             for x in rows)
        gain = [x for x in c if x['rupees'] > 0][:3]
        loss = [x for x in reversed(c) if x['rupees'] < 0][:3]
        items.append("<b>Who made and lost the money:</b> "
                     + (f"made: {names(gain)}. " if gain else "")
                     + (f"Lost: {names(loss)}." if loss else ""))
    out = (_h("Portfolio X-ray: how risky is what we hold?")
           + "<ul style='font-size:14px;line-height:1.6'>"
           + "".join(f"<li>{t}</li>" for t in items) + "</ul>")
    if r.get('short_history'):
        out += _note("Left out for too little price history: "
                     + ", ".join(html.escape(s) for s in r['short_history']) + ".")
    return out + _note(f"Worked out from how today's holdings moved over the last "
                       f"{r.get('days_used', 252)} trading days. The past is a "
                       "guide, not a promise.")


def filings_html(d: Dict) -> str:
    f = d.get('filings') or {}
    news, coming = f.get('news') or [], f.get('upcoming') or []
    if not news and not coming:
        return ""
    out = _h("Company news on our stocks (official NSE filings)")
    if coming:
        out += ("<p style='font-size:14px;margin:4px 0;color:#8a6d00'>📅 Coming up: "
                + "; ".join(
                    f"<b>{html.escape(x['symbol'])}</b> {html.escape(x['what'])} on "
                    f"{date.fromisoformat(x['date']):%a %d %b}" for x in coming)
                + ".</p>")
    if news:
        rows = []
        for x in news:
            text = html.escape(x['text'])
            if x.get('link'):
                text = f"<a href='{html.escape(x['link'], quote=True)}'>{text}</a>"
            rows.append((date.fromisoformat(x['date']).strftime('%a %d %b'),
                         f"<b>{html.escape(x['symbol'])}</b>", text))
        out += _table([('Day', 'left'), ('Stock', 'left'), ('Filing', 'left')], rows)
    return out


# ---------------------------------------------------------------------------
# Signal report card
# ---------------------------------------------------------------------------
def report_card(shadow: Dict, voting: List[str]) -> List[Dict]:
    """One row per stock-picking signal: how its live grade stands against
    the bar for earning a vote (signal_decay_detector's promotion rule)."""
    rows = []
    for key, name in SIGNAL_NAMES.items():
        g = (shadow.get('rank_ic') or {}).get(key) or {}
        s = g.get('20d') or {}
        ic, t, n = s.get('ic'), s.get('t_stat'), s.get('n_dates') or 0
        cover = g.get('coverage_pct')
        if key in voting:
            status, color = 'Has the vote (chosen by the ten-year test)', GREEN
        elif n == 0:
            status = ('Waiting for data' if not cover
                      else 'Just started: first grade after about 20 trading days')
            color = GREY
        elif n >= PROMOTE_MIN_DAYS and (ic or 0) >= PROMOTE_MIN_IC and (t or 0) >= PROMOTE_MIN_T:
            status, color = 'Passed: earns a vote', GREEN
        elif n >= 60 and (ic or 0) < 0:
            status, color = 'Failing: its picks did worse, not better', RED
        else:
            status = f"On probation: {n} of {PROMOTE_MIN_DAYS} days graded"
            color = AMBER
        if ic is None:
            verdict = 'no grade yet'
        else:
            verdict = ('helping' if ic >= PROMOTE_MIN_IC else 'hurting'
                       if ic <= -0.01 else 'no clear effect')
            if abs(t or 0) < PROMOTE_MIN_T:
                verdict += ' (could be luck)'
        rows.append({'key': key, 'name': name, 'status': status, 'color': color,
                     'verdict': verdict, 'ic': ic, 'days': n, 'coverage': cover})
    return rows


def report_card_html(shadow: Dict, voting: List[str]) -> str:
    rows = report_card(shadow, voting)
    if not rows:
        return ""
    return (_h("Signal report card: which ideas are earning their place?")
            + _table([('Signal', 'left'), ('So far', 'left'), ('Status', 'left')],
                     [(html.escape(r['name']), r['verdict'],
                       f"<span style='color:{r['color']}'>{r['status']}</span>")
                      for r in rows])
            + _note(f"Each signal ranks every stock daily. We then check whether "
                    f"its favourites really did better over the next 20 days. A "
                    f"signal gets a vote only after {PROMOTE_MIN_DAYS} graded days "
                    "of clearly helping, and loses it if it starts hurting."))


# ---------------------------------------------------------------------------
# Chart
# ---------------------------------------------------------------------------
def make_dashboard_chart(d: Dict, path: Path = CHART_FILE) -> bool:
    """Six small charts in one picture. False when matplotlib is missing or
    there is nothing to draw."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    def xy(s: Optional[Dict]):
        if not s or len(s.get('values', [])) < 2:
            return None
        return [date.fromisoformat(x) for x in s['dates']], s['values']

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), dpi=110)
    drawn = 0

    import matplotlib.dates as mdates

    def style(ax, title):
        ax.set_title(title, fontsize=10, loc='left', color=NAVY, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7)
        if ax.lines:
            loc = mdates.AutoDateLocator(maxticks=6)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))

    ax = axes[0][0]
    n, m = xy(d.get('series_nifty')), xy(d.get('series_nifty_200dma'))
    if n:
        ax.plot(*n, color=NAVY, linewidth=1.8, label='NIFTY 50')
        if m:
            ax.plot(*m, color=ORANGE, linewidth=1.4, label='200-day average')
        ax.legend(frameon=False, fontsize=8)
        drawn += 1
    style(ax, '1. Trend: NIFTY and its 200-day average')

    ax = axes[0][1]
    b = xy((d.get('breadth') or {}).get('series'))
    if b:
        ax.plot(*b, color=NAVY, linewidth=1.8)
        ax.axhline(60, color=GREEN, linewidth=0.8, linestyle='--')
        ax.axhline(40, color=RED, linewidth=0.8, linestyle='--')
        ax.set_ylim(0, 100)
        drawn += 1
    style(ax, '2. Breadth: % of stocks in an uptrend')

    ax = axes[0][2]
    v = (d.get('valuation') or {})
    p = xy(v.get('series'))
    if p:
        ax.plot(*p, color=NAVY, linewidth=1.8)
        med = (v.get('NIFTY 50') or {}).get('pe_median')
        if med:
            ax.axhline(med, color=GREY, linewidth=0.9, linestyle='--')
            ax.text(p[0][0], med, ' usual', fontsize=7, color=GREY, va='bottom')
        ax.scatter([p[0][-1]], [p[1][-1]], color=ORANGE, zorder=3, s=28)
        drawn += 1
    style(ax, '3. Price tag: NIFTY price ÷ profits (P/E)')

    ax = axes[1][0]
    x = xy((d.get('vix') or {}).get('series'))
    if x:
        ax.plot(*x, color=NAVY, linewidth=1.8)
        ax.axhline(15, color=GREEN, linewidth=0.8, linestyle='--')
        ax.axhline(20, color=RED, linewidth=0.8, linestyle='--')
        drawn += 1
    style(ax, '4. Fear gauge: India VIX')

    ax = axes[1][1]
    f = d.get('flows') or {}
    fi, di = xy(f.get('series_fii')), xy(f.get('series_dii'))
    if fi:
        ax.plot(*fi, color=NAVY, linewidth=1.8, label='Foreign investors')
        if di:
            ax.plot(*di, color=ORANGE, linewidth=1.6, label='Indian funds')
        ax.axhline(0, color='#cccccc', linewidth=0.8)
        ax.legend(frameon=False, fontsize=8)
        drawn += 1
    style(ax, '5. Big money: net buying, last 100 days (₹ crore)')

    ax = axes[1][2]
    sec = [r for r in (d.get('sectors') or [])
           if r['group'] == 'sector' and r['month'] is not None]
    if sec:
        sec = sorted(sec, key=lambda r: r['month'])
        ax.barh([r['name'] for r in sec], [r['month'] for r in sec],
                color=[GREEN if r['month'] >= 0 else RED for r in sec])
        ax.axvline(0, color='#888888', linewidth=0.8)
        drawn += 1
    ax.set_title('6. Sectors: change over 1 month (%)', fontsize=10, loc='left',
                 color=NAVY, fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=7)

    if not drawn:
        plt.close(fig)
        return False
    fig.suptitle(f"Market dashboard, {d.get('as_of', '')}", fontsize=12,
                 x=0.01, ha='left', color=NAVY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path)
    plt.close(fig)
    return True
