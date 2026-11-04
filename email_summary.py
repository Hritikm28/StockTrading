"""
Build the daily email summary (HTML body + subject) from today's run.
Writes:  email_body.html, email_subject.txt
Used by the GitHub Actions daily workflow to email the daily summary.

The headline is the paper portfolio (portfolio_book.py): what it holds,
the orders for the next open, and how it is doing against NIFTY.
"""
import json
from datetime import date
from pathlib import Path

today = date.today().strftime('%Y-%m-%d')
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
    live_alpha_str = ", ".join(sorted(get_live_alphas()))
except Exception:
    pass

REASONS = {'rank_dropped': 'fell out of the top 30 at the monthly review',
           'trend_exit': 'NIFTY fell 2% below its 200-day average',
           'top_rank': 'top-ranked, fills an empty slot'}


def status_banner():
    items = []
    if not gate_open:
        v = f" (NIFTY {vs200:+.1f}% vs 200-DMA)" if vs200 is not None else ""
        held = (" Current holdings are kept unless NIFTY falls 2% below it."
                if book.get('n_holdings') else "")
        items.append(("#b0322b", f"⛔ TREND GATE CLOSED{v}: no new buys until "
                                 f"NIFTY reclaims its 200-day average.{held}"))
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
    perf = (f"Since {book['inception']}: portfolio {_pct(book.get('return_pct'))}"
            f" vs NIFTY {_pct(book.get('nifty_return_pct'))}"
            f" (value ₹{book['nav']:,.0f}, costs included)")
    buy_rows = [(f"<b>{o['symbol']}</b>", o.get('rank', ''),
                 _fmt(o.get('score')), f"₹{o['value']:,.0f}") for o in buys]
    sell_rows = [(f"<b>{o['symbol']}</b>", REASONS.get(o.get('reason'),
                                                       o.get('reason', '')))
                 for o in sells]
    hold_rows = [(f"<b>{h['symbol']}</b>", h['entry_date'],
                  _fmt(h['entry_price']), _fmt(h['last_price']),
                  _pct(h['pnl_pct']), f"{h['weight_pct']:.1f}%")
                 for h in book.get('holdings', [])]
    out = (f"<p style='font-size:14px;margin:4px 0'>{status}<br>{perf}</p>")
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
  stocks, replaced only at the monthly review, 0.4% round-trip costs
  charged. This is a research experiment, not investment advice.
</p>
</body></html>"""

Path('email_body.html').write_text(html, encoding='utf-8')
flag = ""
if breaker_tripped:
    flag = "[BREAKER] "
elif not gate_open:
    flag = "[GATE CLOSED] "
held = book.get('n_holdings', 0)
subject = (f"{flag}Portfolio {today}: {len(buys)} buy / {len(sells)} sell, "
           f"{held} held [{regime}]")
Path('email_subject.txt').write_text(subject, encoding='utf-8')
print("Wrote email_body.html and email_subject.txt")
print("Subject:", subject)
