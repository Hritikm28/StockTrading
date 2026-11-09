"""
Build the daily email summary (HTML body + subject) from today's run.
Writes:  email_body.html, email_subject.txt
Used by the GitHub Actions daily workflow to email the daily summary.

The headline is the paper portfolio (portfolio_book.py): what it holds,
the orders for the next open, and how it is doing against NIFTY.
"""
import json
from datetime import date

from market_calendar import latest_session
from pathlib import Path

today = latest_session().strftime('%Y-%m-%d')   # same session as daily_runner
sig_dir = Path('signals')

# Today's run report (written by daily_runner, published by publish_signals)
report = {}
rj = sig_dir / f'{today}_summary.json'
if not rj.exists():
    rj = Path('daily_reports') / f'report_{today}.json'
if rj.exists():
    try:
        report = json.loads(rj.read_text(encoding='utf-8'))
    except Exception:
        report = {}
regime = report.get('regime', 'UNKNOWN')
vix = report.get('vix', '')
gate_open = report.get('trend_gate_open', True)
vs200 = report.get('nifty_vs_200dma')
breaker_tripped = report.get('breaker_tripped', False)
if not report:
    try:
        cb = json.loads(Path('paper_trading/circuit_breaker.json')
                        .read_text(encoding='utf-8'))
        breaker_tripped = bool(cb.get('tripped', False))
    except Exception:
        pass

book = report.get('portfolio') or {}
orders = book.get('orders', [])
buys = [o for o in orders if o.get('side') == 'BUY']
sells = [o for o in orders if o.get('side') == 'SELL']

live_alpha_str = ""
try:
    import sys
    sys.path.insert(0, '.')
    from signal_decay_detector import get_live_alphas
    _names = {'resid_mom': 'residual momentum', 'momentum': 'momentum',
              'mom_riskadj': 'risk-adjusted momentum'}
    live_alpha_str = ", ".join(sorted(_names.get(a, a) for a in get_live_alphas()))
except Exception:
    pass

REASONS = {'rank_dropped': 'fell out of the top 30 at the monthly review',
           'trend_exit': 'NIFTY fell 2% below its 200-day average',
           'trend_cut': 'NIFTY fell 2% below its 200-day average: '
                        'selling half of every holding',
           'trend_restored': 'NIFTY is back above its 200-day average: '
                             'buying back to full size',
           'top_rank': 'top-ranked, fills an empty slot'}
exposure = book.get('exposure_pct')


def status_banner():
    items = []
    v = f" (NIFTY {vs200:+.1f}% vs 200-DMA)" if vs200 is not None else ""
    if exposure is not None and exposure < 100:
        items.append(("#8a6d00", f"⚠️ MARKET TREND DOWN{v}: staying {exposure}% "
                                 f"invested until NIFTY climbs back above its "
                                 f"200-day average."))
    elif not gate_open:
        items.append(("#8a6d00", f"⚠️ NIFTY just below its 200-day average{v}: "
                                 f"holding, no new buys. Holdings are cut to "
                                 f"half if it falls 2% below."))
    elif vs200 is not None:
        items.append(("#1a7a3a", f"✅ Trend gate open (NIFTY {vs200:+.1f}% "
                                 "vs 200-DMA)"))
    if breaker_tripped:
        items.append(("#b0322b", "⛔ CIRCUIT BREAKER TRIPPED — recent portfolio "
                                 "stats breached limits; no new buys until reset"))
    if live_alpha_str:
        items.append(("#555", f"Signals with a vote: <b>{live_alpha_str}</b>"
                              " — all others on probation (graded, no vote)"))
    if not items:
        return ""
    rows_html = "".join(
        f"<div style='color:{c};font-size:13px;margin:2px 0'>{t}</div>"
        for c, t in items)
    return (f"<div style='background:#f4f7fb;border-left:4px solid #1a2b4a;"
            f"padding:8px 12px;margin:8px 0'>{rows_html}</div>")


def _fmt(v, nd=2):
    try:
        return f"{float(v):,.{nd}f}"
    except Exception:
        return str(v)


def _table(headers, rows):
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
            f"<tr style='background:#1a2b4a;color:#fff'>{head}</tr>{body}</table>")


def _pct(v):
    if v is None:
        return "n/a"
    color = '#1a7a3a' if v > 0 else '#b0322b' if v < 0 else '#555'
    return f"<b style='color:{color}'>{v:+.2f}%</b>"


def portfolio_section():
    if not book:
        return "<p style='color:#888'>Portfolio not updated today.</p>"
    status = (f"Holding <b>{book['n_holdings']}</b> of "
              f"{book['target_holdings']} stocks, {book['cash_pct']:.0f}% cash"
              if book['n_holdings'] else "<b>All in cash</b>")
    if exposure is not None:
        status += f" (target {exposure}% invested)"
    perf = (f"Since {book['inception']}: portfolio {_pct(book.get('return_pct'))}"
            f" vs NIFTY {_pct(book.get('nifty_return_pct'))}")
    if book.get('momfund_return_pct') is not None:
        perf += (f" vs the {book.get('momfund_name', 'momentum')} index fund "
                 f"{_pct(book.get('momfund_return_pct'))}")
    perf += f" (value ₹{book['nav']:,.0f}, costs included)"
    sold = book.get('if_sold_today')
    if sold and book['n_holdings']:
        perf += (f"<br>If you sold everything today: ₹{sold['after_costs']:,.0f}"
                 f" after selling costs, "
                 + (f"estimated tax ₹{sold['tax']:,.0f}, so you'd keep "
                    f"<b>₹{sold['after_tax']:,.0f}</b>." if sold['tax'] > 0
                    else "no tax (no profit this financial year)."))
    buy_rows = [(f"<b>{o['symbol']}</b>",
                 'add' if o.get('topup') else o.get('rank', ''),
                 '' if o.get('topup') else _fmt(o.get('score')),
                 f"₹{o['value']:,.0f}") for o in buys]
    sell_rows = [(f"<b>{o['symbol']}</b>",
                  ("half — " if o.get('fraction', 1) < 1 else "")
                  + REASONS.get(o.get('reason'), o.get('reason', '')))
                 for o in sells]
    hold_rows = [(f"<b>{h['symbol']}</b>", h['entry_date'],
                  _fmt(h['entry_price']), _fmt(h['last_price']),
                  _pct(h['pnl_pct']), f"{h['weight_pct']:.1f}%")
                 for h in book.get('holdings', [])]
    out = (f"<p style='font-size:14px;margin:4px 0'>{status}<br>{perf}</p>")
    out += data_check_line(book.get('data_check'))
    soon = book.get('results_soon') or []
    if soon:
        out += ("<p style='font-size:13px;margin:4px 0;color:#8a6d00'>📅 Results "
                "coming (where known): " + ", ".join(
                    f"<b>{r['symbol'].replace('.NS', '')}</b> "
                    f"{date.fromisoformat(r['date']):%a %d %b}" for r in soon)
                + ". Prices often jump or drop on results day.</p>")
    if buys or sells:
        out += "<h3 style='margin-bottom:4px'>Orders for the next market open</h3>"
        if buys:
            out += ("<p style='margin:4px 0;color:#1a7a3a'><b>BUY</b></p>"
                    + _table([('Symbol', 'left'), ('Rank', 'center'),
                              ('Score', 'center'), ('Amount', 'right')],
                             buy_rows))
        if sells:
            out += ("<p style='margin:4px 0;color:#b0322b'><b>SELL</b></p>"
                    + _table([('Symbol', 'left'), ('Why', 'left')], sell_rows))
        out += copy_orders(book.get('copy'))
    else:
        out += ("<p style='font-size:13px;color:#555'>No orders today. Names "
                "are replaced only at the monthly review, so most days are "
                "quiet by design.</p>")
    if hold_rows:
        out += ("<h3 style='margin-bottom:4px'>Current holdings</h3>"
                + _table([('Symbol', 'left'), ('Bought', 'center'),
                          ('Entry', 'right'), ('Last', 'right'),
                          ('P&L', 'right'), ('Weight', 'right')], hold_rows))
    return out


def data_check_line(dc):
    if not dc or not dc.get('prices_as_of'):
        return ""
    day = date.fromisoformat(dc['prices_as_of']).strftime('%a %d %b')
    if dc.get('ok'):
        return (f"<p style='font-size:13px;margin:4px 0;color:#1a7a3a'>✅ Data "
                f"check: every price is up to date (closing prices of {day}).</p>")
    bad = dc.get('stale_benchmarks', []) + dc.get('stale_stocks', [])
    return (f"<p style='font-size:13px;margin:4px 0;color:#b0322b'>⚠️ Data "
            f"check: these prices are older than {day}: <b>{', '.join(bad)}</b>. "
            f"Figures that use them may be off until they update.</p>")


def copy_orders(plan):
    """Whole shares for someone copying the book with plan['capital']."""
    if not plan or not plan.get('orders'):
        return ""
    rows = [(f"<b>{r['symbol'].replace('.NS', '')}</b>",
             'add' if r['topup'] else 'new', _fmt(r['price']),
             f"{r['shares']:,}" if r['shares'] else "— (too expensive)",
             f"₹{r['cost']:,.0f}") for r in plan['orders']]
    total = sum(r['cost'] for r in plan['orders'])
    return (f"<p style='margin:8px 0 4px;font-size:13px'><b>If you're copying "
            f"with ₹{plan['capital']:,.0f}</b> (whole shares, at the last "
            f"close; the open may differ):</p>"
            + _table([('Stock', 'left'), ('', 'center'), ('Price', 'right'),
                      ('Shares', 'right'), ('Cost', 'right')], rows)
            + f"<p style='font-size:12px;color:#555;margin:4px 0'>Total about "
              f"₹{total:,.0f}. To hold at least one share of every stock you "
              f"need about ₹{plan['min_capital']:,.0f}.</p>")


def scoreboard():
    """Honest track record section: net of costs, vs NIFTY benchmark."""
    sj = Path('paper_trading/results/summary.json')
    if not sj.exists():
        return ""
    try:
        data = json.loads(sj.read_text(encoding='utf-8'))
    except Exception:
        return ""
    metrics = data.get('metrics', {})
    track = data.get('track_record', {})
    cost = data.get('cost_model_pct', 0.4)

    rows = ""
    for w in ['1d', '3d', '5d', '10d']:
        m = metrics.get(w)
        if not m:
            continue
        hl = " style='background:#eef6ee'" if w == '5d' else ""
        exc = m.get('avg_excess')
        exc_str = f"{exc:+.2f}%" if exc is not None else "n/a"
        exc_color = '#1a7a3a' if (exc or 0) > 0 else '#b0322b'
        wlabel = f"{w} ★" if w == '5d' else w
        rows += (
            f"<tr{hl}><td>{wlabel}</td><td align='center'>{m['n_trades']}</td>"
            f"<td align='center'>{m['win_rate']:.0f}%</td>"
            f"<td align='center'>{m['avg_ret']:+.2f}%</td>"
            f"<td align='center' style='color:{exc_color}'><b>{exc_str}</b></td></tr>"
        )
    if not rows:
        return ""

    cum = ""
    if track:
        exc_c = track.get('cum_excess_pct', 0)
        color = '#1a7a3a' if exc_c > 0 else '#b0322b'
        cum = (
            f"<p style='font-size:13px;margin:6px 0'>Since {track.get('inception','')} "
            f"({track.get('days_live',0)} signal days, {track.get('total_trades',0)} trades): "
            f"strategy <b>{track.get('cum_net_pct',0):+.2f}%</b> vs NIFTY "
            f"<b>{track.get('cum_nifty_pct',0):+.2f}%</b> → excess "
            f"<b style='color:{color}'>{exc_c:+.2f}%</b></p>"
        )

    return (
        f"<h3 style='margin-bottom:4px'>Old signal-by-signal record "
        f"<span style='font-weight:normal;font-size:12px;color:#888'>"
        f"(before the portfolio; net of {cost:.2f}% costs, vs NIFTY)</span></h3>"
        f"{cum}"
        f"<table cellpadding='5' cellspacing='0' "
        f"style='border-collapse:collapse;font-size:13px'>"
        f"<tr style='background:#1a2b4a;color:#fff'><th>Window</th><th>Trades</th>"
        f"<th>Win rate</th><th>Avg net ret</th><th>Avg excess vs NIFTY</th></tr>"
        f"{rows}</table>"
    )


def shadow_grade():
    """Grade of the untraded picks (shadow_grader.py): the live record only
    covers trades the gate allowed, so this is the honest edge check."""
    sj = Path('paper_trading/results/shadow_report.json')
    if not sj.exists():
        return ""
    try:
        r = json.loads(sj.read_text(encoding='utf-8'))
    except Exception:
        return ""
    color = {'EDGE': '#1a7a3a', 'NO_EDGE': '#b0322b'}.get(r.get('verdict'), '#8a6d00')
    sp = r.get('shadow_portfolio', {})
    line = ""
    if sp.get('top_picks_net_pct') is not None:
        line = (f"<p style='font-size:13px;margin:4px 0'>If every top pick since "
                f"{r.get('since','')} had been traded: <b>{sp['top_picks_net_pct']:+.2f}%</b> "
                f"vs NIFTY same timing <b>{sp.get('nifty_same_timing_pct') or 0:+.2f}%</b> "
                f"(gated live system: {sp.get('live_gated_pct') or 0:+.2f}%)</p>")
    return (f"<h3 style='margin-bottom:4px'>Shadow grade — do the picks have an edge?</h3>"
            f"<p style='font-size:13px;margin:4px 0'><b style='color:{color}'>"
            f"{r.get('verdict','')}</b>: {r.get('verdict_text','')}</p>{line}")


html = f"""<html><body style='font-family:Segoe UI,Arial,sans-serif;color:#222'>
<h2 style='margin-bottom:4px'>Daily Portfolio — {today}</h2>
<p style='margin-top:0;color:#555'>
  Market regime: <b>{regime}</b>{f" &nbsp;|&nbsp; VIX: <b>{vix}</b>" if vix else ""}
  &nbsp;|&nbsp; {len(buys)} buy / {len(sells)} sell order(s)
</p>
{status_banner()}
{portfolio_section()}
{shadow_grade()}
{scoreboard()}
<p style='font-size:12px;color:#999;margin-top:20px'>
  Paper portfolio, auto-generated on GitHub Actions after market close from
  today's closing data. Orders are for the next market open. Up to 15
  stocks, replaced only at the monthly review; half invested while the
  market trend is down; 0.4% round-trip costs charged. This is a research
  experiment, not investment advice.
</p>
</body></html>"""

Path('email_body.html').write_text(html, encoding='utf-8')
flag = ""
if breaker_tripped:
    flag = "[BREAKER] "
elif exposure is not None and exposure < 100:
    flag = f"[TREND DOWN: {exposure}% INVESTED] "
elif not gate_open:
    flag = "[BELOW 200-DMA] "
held = book.get('n_holdings', 0)
subject = (f"{flag}Portfolio {today}: {len(buys)} buy / {len(sells)} sell, "
           f"{held} held [{regime}]")
Path('email_subject.txt').write_text(subject, encoding='utf-8')
print("Wrote email_body.html and email_subject.txt")
print("Subject:", subject)
