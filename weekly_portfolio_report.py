"""
Weekly portfolio report, emailed on Sundays by weekly_review.yml.

Replaces the "Weekly Correction Report", which graded the old signal-by-
signal system and a per-stock dampener retired in June 2026. This one
reports on what actually runs: the 15-stock paper portfolio.

  1. This week and since the start, against NIFTY and a momentum index
     fund, in plain words, and what you'd keep after costs and tax
  2. A data check, so wrong numbers never pass silently
  3. Day by day, plus a chart (attached as weekly_chart.png)
  4. Every holding, best to worst, and results due soon
  5. Trades made this week
  6. What happens next (pending orders, trend rule, next monthly review)
  7. Whole-share copy list for a chosen amount
  8. Whether the stock picking has shown an edge yet

Reads only committed files (paper_trading/portfolio.json, the latest
signals/*_summary.json, the shadow report). Writes weekly_email_body.html,
weekly_email_subject.txt and weekly_chart.png.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from costs import sell_cost
from portfolio_book import (BOOK_FILE, EXIT_BAND_PCT, MOMFUND_NAME,
                            copy_plan, tax_estimate)

SHADOW_FILE = Path('paper_trading/results/shadow_report.json')
SIGNALS_DIR = Path('signals')
BODY_FILE = Path('weekly_email_body.html')
SUBJECT_FILE = Path('weekly_email_subject.txt')
CHART_FILE = Path('weekly_chart.png')

REASONS = {'top_rank': 'new pick',
           'trend_restored': 'buy more (trend back up)',
           'trend_cut': 'sell half (trend down)',
           'trend_exit': 'sell (trend down)',
           'rank_dropped': 'dropped out of the top 30'}
GREEN, RED, NAVY = '#1a7a3a', '#b0322b', '#1a2b4a'


def _chg(a, b) -> Optional[float]:
    if not a or b is None:
        return None
    return round((b / a - 1) * 100, 2)


def _pct(v) -> str:
    if v is None:
        return 'n/a'
    color = GREEN if v > 0 else RED if v < 0 else '#555'
    return f"<span style='color:{color}'>{v:+.2f}%</span>"


def _inr(v) -> str:
    return f"₹{v:,.0f}"


def _table(headers, rows) -> str:
    if not rows:
        return "<p style='color:#888'>None</p>"
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


def _name(sym: str) -> str:
    return sym.replace('.NS', '')


def week_window(nav: List[Dict]) -> Tuple[Dict, List[Dict]]:
    """(reference point, this week's points). The week runs from Monday of
    the latest point's week; the reference is the last point before that
    Monday, or the first point if the portfolio started this week."""
    last_day = date.fromisoformat(nav[-1]['date'])
    monday = (last_day - timedelta(days=last_day.weekday())).isoformat()
    before = [p for p in nav if p['date'] < monday]
    return (before[-1] if before else nav[0],
            [p for p in nav if p['date'] >= monday])


def holdings_rows(book: Dict) -> List[Dict]:
    """Each holding's gain since bought, after costs (as the daily email)."""
    nav = book['nav'][-1]['nav'] if book.get('nav') else None
    out = []
    for sym, h in book.get('holdings', {}).items():
        value = h['shares'] * h['last_price']
        out.append({'symbol': sym, 'entry_date': h['entry_date'],
                    'pnl_pct': round(((value - sell_cost(value))
                                      / h['cost_basis'] - 1) * 100, 2),
                    'value': value,
                    'weight_pct': round(value / nav * 100, 1) if nav else None})
    return sorted(out, key=lambda r: -r['pnl_pct'])


def next_review(last_rebalance: Optional[str], today: date) -> str:
    """The monthly review runs on the first trading day of a new month."""
    if last_rebalance:
        y, m = map(int, last_rebalance.split('-'))
    else:
        y, m = today.year, today.month
    y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return date(y, m, 1).strftime('%B %Y')


def build(book: Dict, summary: Dict, shadow: Dict) -> Tuple[str, str]:
    nav = sorted(book.get('nav', []), key=lambda p: p['date'])
    if not nav:
        return ("Weekly portfolio: not started yet",
                "<p>The paper portfolio has no history yet.</p>")
    last = nav[-1]
    last_day = date.fromisoformat(last['date'])
    ref, week = week_window(nav)
    port_w, nifty_w = _chg(ref['nav'], last['nav']), _chg(ref.get('nifty'), last.get('nifty'))
    port_all = _chg(nav[0]['nav'], last['nav'])
    nifty_all = _chg(nav[0].get('nifty'), last.get('nifty'))
    mom_w = _chg(ref.get('momfund'), last.get('momfund'))
    mom_all = _chg(nav[0].get('momfund'), last.get('momfund'))
    daily = summary.get('portfolio') or {}

    def verdict(p, n):
        if p is None or n is None:
            return ''
        gap = round(p - n, 2)
        if gap > 0:
            return f"<b style='color:{GREEN}'>ahead of NIFTY by {gap:.2f}%</b>"
        if gap < 0:
            return f"<b style='color:{RED}'>behind NIFTY by {-gap:.2f}%</b>"
        return "<b>level with NIFTY</b>"

    def fund(v):
        return (f"; the {MOMFUND_NAME} index fund {_pct(v)}"
                if v is not None else "")

    # 1. Headline
    sold = tax_estimate(book, last_day) if book.get('holdings') else None
    keep = ""
    if sold:
        keep = (f"<br>If you sold everything today: {_inr(sold['after_costs'])} "
                "after selling costs, "
                + (f"estimated tax {_inr(sold['tax'])}, so you'd keep "
                   f"<b>{_inr(sold['after_tax'])}</b>." if sold['tax'] > 0
                   else "and no tax to pay (no profit this financial year)."))
    html = [f"<h2 style='color:{NAVY};margin-bottom:4px'>Weekly portfolio — "
            f"week to {last_day:%a %d %b %Y}</h2>",
            "<p style='font-size:15px;line-height:1.6'>"
            f"<b>This week:</b> portfolio {_pct(port_w)} vs NIFTY {_pct(nifty_w)}"
            f" — {verdict(port_w, nifty_w)}{fund(mom_w)}<br>"
            f"<b>Since the start ({nav[0]['date']}):</b> portfolio "
            f"{_pct(port_all)} vs NIFTY {_pct(nifty_all)} — "
            f"{verdict(port_all, nifty_all)}{fund(mom_all)}<br>"
            f"Value {_inr(last['nav'])} (started at {_inr(nav[0]['nav'])}), "
            f"cash {_inr(last['cash'])}, {last['n_holdings']} stocks held. "
            f"Costs included.{keep}</p>"]
    if mom_all is None:
        html.append("<p style='font-size:12px;color:#555;margin:0'>The momentum "
                    "fund comparison starts once its prices are collected "
                    "daily.</p>")
    dc = daily.get('data_check')
    if dc and dc.get('prices_as_of'):
        day = date.fromisoformat(dc['prices_as_of']).strftime('%a %d %b')
        bad = dc.get('stale_benchmarks', []) + dc.get('stale_stocks', [])
        html.append(
            f"<p style='font-size:13px;color:{GREEN}'>✅ Data check: every price "
            f"is up to date (closing prices of {day}).</p>" if dc.get('ok') else
            f"<p style='font-size:13px;color:{RED}'>⚠️ Data check: these prices "
            f"are older than {day}: <b>{', '.join(bad)}</b>. Figures that use "
            "them may be off.</p>")

    # 2. Day by day
    rows, prev = [], ref
    for p in week:
        if p is ref:
            continue
        rows.append((date.fromisoformat(p['date']).strftime('%a %d %b'),
                     _inr(p['nav']), _pct(_chg(prev['nav'], p['nav'])),
                     _pct(_chg(prev.get('nifty'), p.get('nifty'))),
                     _pct(_chg(prev.get('momfund'), p.get('momfund')))
                     if p.get('momfund') and prev.get('momfund') else '—'))
        prev = p
    html += ["<h3 style='margin-bottom:4px'>Day by day</h3>",
             _table([('Day', 'left'), ('Value', 'right'),
                     ('Portfolio', 'right'), ('NIFTY', 'right'),
                     ('Momentum fund', 'right')], rows),
             "<p style='font-size:12px;color:#555;margin:4px 0'>The chart "
             "attached to this email shows the whole run.</p>"]

    # 3. Holdings, best to worst
    hold = holdings_rows(book)
    if hold:
        best, worst = hold[0], hold[-1]
        html.append("<h3 style='margin-bottom:4px'>Our stocks, best to worst</h3>"
                    f"<p style='margin:4px 0'>Best: <b>{_name(best['symbol'])}</b> "
                    f"{_pct(best['pnl_pct'])} · Worst: <b>{_name(worst['symbol'])}"
                    f"</b> {_pct(worst['pnl_pct'])} (since bought, after costs)</p>")
        html.append(_table(
            [('Stock', 'left'), ('Bought', 'center'), ('Since bought', 'right'),
             ('Value', 'right'), ('Weight', 'right')],
            [(f"<b>{_name(r['symbol'])}</b>", r['entry_date'], _pct(r['pnl_pct']),
              _inr(r['value']),
              f"{r['weight_pct']:.1f}%" if r['weight_pct'] is not None else '')
             for r in hold]))
    soon = daily.get('results_soon') or []
    if soon:
        html.append("<p style='font-size:13px;color:#8a6d00'>📅 Results coming "
                    "(where known): " + ", ".join(
                        f"<b>{_name(r['symbol'])}</b> "
                        f"{date.fromisoformat(r['date']):%a %d %b}" for r in soon)
                    + ". Prices often jump or drop on results day.</p>")

    # 4. Trades this week
    monday = (last_day - timedelta(days=last_day.weekday())).isoformat()
    trades = [t for t in book.get('trades', [])
              if monday <= t['date'] <= last['date']]
    html += ["<h3 style='margin-bottom:4px'>Trades this week</h3>",
             _table([('Day', 'left'), ('Stock', 'left'), ('Action', 'left'),
                     ('Price', 'right'), ('Amount', 'right')],
                    [(date.fromisoformat(t['date']).strftime('%a %d %b'),
                      _name(t['symbol']),
                      f"{t['side'].title()} — {REASONS.get(t.get('reason'), t.get('reason') or '')}"
                      + (" (half)" if t.get('fraction', 1) < 1 else ""),
                      f"{t['price']:,.2f}", _inr(t['value'])) for t in trades])]

    # 5. What happens next
    nxt = []
    pending = book.get('pending', [])
    if pending:
        buys = [o for o in pending if o['side'] == 'BUY']
        sells = [o for o in pending if o['side'] == 'SELL']
        bits = []
        if buys:
            bits.append(f"buy {len(buys)} ({_inr(sum(o['value'] for o in buys))})")
        if sells:
            bits.append(f"sell {len(sells)}")
        why = sorted({REASONS.get(o.get('reason'), o.get('reason', '')) for o in pending})
        nxt.append(f"Orders for the next market open: {' and '.join(bits)} — "
                   f"{', '.join(why)}.")
    else:
        nxt.append("No orders waiting. Most weeks are quiet by design.")
    vs200 = summary.get('nifty_vs_200dma')
    exposure = book.get('exposure')
    if exposure is not None and exposure < 1.0:
        nxt.append(f"Trend rule: NIFTY is below its 200-day average"
                   + (f" ({vs200:+.1f}%)" if vs200 is not None else "")
                   + f", so we stay {exposure:.0%} invested until it climbs "
                     "back above.")
    elif vs200 is not None:
        nxt.append(f"Trend rule: NIFTY is {vs200:+.1f}% vs its 200-day average,"
                   " so we stay fully invested. If it falls "
                   f"{EXIT_BAND_PCT:.0f}% below, we sell half of every stock.")
    nxt.append(f"Next stock review: first trading day of "
               f"{next_review(book.get('last_rebalance'), last_day)}. Stocks "
               "that drop out of the top 30 are replaced; the rest are kept.")
    html += ["<h3 style='margin-bottom:4px'>What happens next</h3>",
             "<ul style='font-size:14px;line-height:1.6'>"
             + "".join(f"<li>{t}</li>" for t in nxt) + "</ul>"]

    # 7. Copy list
    plan = copy_plan(book, [], last['nav'])
    if plan['holdings']:
        rows = [(f"<b>{_name(r['symbol'])}</b>", f"{r['price']:,.2f}",
                 f"{r['shares']:,}" if r['shares'] else "— (too expensive)",
                 _inr(r['cost'])) for r in plan['holdings']]
        html += [f"<h3 style='margin-bottom:4px'>If you copied it with "
                 f"{_inr(plan['capital'])}</h3>",
                 "<p style='font-size:13px;margin:4px 0'>Whole shares at "
                 "Friday's closing prices, same weights as the portfolio:</p>",
                 _table([('Stock', 'left'), ('Price', 'right'),
                         ('Shares', 'right'), ('Cost', 'right')], rows),
                 f"<p style='font-size:12px;color:#555;margin:4px 0'>That "
                 f"invests about {_inr(plan['invested'])}. To hold at least one "
                 f"share of every stock you'd need about "
                 f"{_inr(plan['min_capital'])}.</p>"]

    # 8. Edge check
    if shadow.get('verdict'):
        label = {'EDGE': ('Yes, so far', GREEN), 'NO_EDGE': ('Not yet', RED),
                 'UNPROVEN': ('Too early to tell', '#8a6d00')
                 }.get(shadow['verdict'], (shadow['verdict'], '#555'))
        html += ["<h3 style='margin-bottom:4px'>Is the stock picking beating the "
                 "market?</h3>",
                 f"<p style='font-size:14px'><b style='color:{label[1]}'>{label[0]}."
                 f"</b> {shadow.get('verdict_text', '')} Judging it fairly needs "
                 "a few months of results, not a few weeks.</p>"]

    html.append("<p style='color:#888;font-size:12px'>Paper portfolio (practice "
                "money), auto-generated weekly from the daily runs. This is a "
                "research experiment, not investment advice.</p>")

    subject = (f"Weekly portfolio, week to {last_day:%d %b}: "
               f"{port_w:+.2f}% vs NIFTY {nifty_w:+.2f}%"
               if port_w is not None and nifty_w is not None
               else f"Weekly portfolio, week to {last_day:%d %b}")
    body = ("<div style='font-family:Arial,sans-serif;max-width:720px'>"
            + "\n".join(html) + "</div>")
    return subject, body


def make_chart(nav: List[Dict], path: Path = CHART_FILE) -> bool:
    """Line chart of the portfolio, NIFTY and the momentum fund, each
    rebased to 100 at the start. Returns False when matplotlib is missing
    or there is too little history."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    nav = sorted(nav, key=lambda p: p['date'])
    if len(nav) < 2:
        return False
    days = [date.fromisoformat(p['date']) for p in nav]

    def rebased(key):
        base = next((p.get(key) for p in nav if p.get(key)), None)
        if not base:
            return None
        return [p[key] / base * 100 if p.get(key) else None for p in nav]

    fig, ax = plt.subplots(figsize=(8, 3.6), dpi=110)
    for key, label, color, width in (('nav', 'Our portfolio', NAVY, 2.4),
                                     ('nifty', 'NIFTY 50', '#888888', 1.6),
                                     ('momfund', f'{MOMFUND_NAME} fund',
                                      '#d08a00', 1.6)):
        ys = rebased(key)
        if ys and sum(y is not None for y in ys) >= 2:
            pts = [(d, y) for d, y in zip(days, ys) if y is not None]
            ax.plot([d for d, _ in pts], [y for _, y in pts], label=label,
                    color=color, linewidth=width)
    ax.axhline(100, color='#cccccc', linewidth=0.8, zorder=0)
    ax.set_title('Growth of ₹100 since the start', fontsize=11, loc='left')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _load(path: Path) -> Dict:
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}


def latest_summary(signals_dir: Path = SIGNALS_DIR) -> Dict:
    files = sorted(signals_dir.glob('20??-??-??_summary.json'))
    return _load(files[-1]) if files else {}


def main():
    book = _load(BOOK_FILE)
    subject, body = build(book, latest_summary(), _load(SHADOW_FILE))
    if make_chart(book.get('nav', [])):
        print(f"Wrote {CHART_FILE}")
    BODY_FILE.write_text(body, encoding='utf-8')
    SUBJECT_FILE.write_text(subject, encoding='utf-8')
    print(f"Wrote {BODY_FILE} and {SUBJECT_FILE}")
    print(f"Subject: {subject}")


if __name__ == '__main__':
    main()
