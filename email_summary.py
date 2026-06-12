"""
Build the daily email summary (HTML body + subject) from today's signals.
Writes:  email_body.html, email_subject.txt
Used by the GitHub Actions daily workflow to email the signal summary.
"""
import csv
import ast
import json
from datetime import date
from pathlib import Path

today = date.today().strftime('%Y-%m-%d')
sig_dir = Path('signals')

# Prefer the dated approved file; fall back to latest.csv
src = sig_dir / f'{today}_approved.csv'
if not src.exists():
    src = sig_dir / 'latest.csv'

rows = []
if src.exists():
    with open(src, newline='') as f:
        rows = list(csv.DictReader(f))

regime = rows[0].get('regime', 'UNKNOWN') if rows else 'UNKNOWN'
vix = rows[0].get('vix', '') if rows else ''
buys = [r for r in rows if r.get('signal') == 'BUY']
sells = [r for r in rows if r.get('signal') == 'SELL']

# System protections status (from the daily report JSON + state files)
report = {}
rj = sig_dir / f'{today}_summary.json'
if not rj.exists():
    rj = Path('daily_reports') / f'report_{today}.json'
if rj.exists():
    try:
        report = json.loads(rj.read_text(encoding='utf-8'))
    except Exception:
        report = {}
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

live_alpha_str = ""
try:
    import sys
    sys.path.insert(0, '.')
    from signal_decay_detector import get_live_alphas
    live_alpha_str = ", ".join(sorted(get_live_alphas()))
except Exception:
    pass


def status_banner():
    items = []
    if not gate_open:
        v = f" (NIFTY {vs200:+.1f}% vs 200-DMA)" if vs200 is not None else ""
        items.append(("#b0322b", f"⛔ TREND GATE CLOSED{v} — no new BUYs; "
                                 "holding cash until NIFTY reclaims its 200-DMA"))
    elif vs200 is not None:
        items.append(("#1a7a3a", f"✅ Trend gate open (NIFTY {vs200:+.1f}% "
                                 "vs 200-DMA)"))
    if breaker_tripped:
        items.append(("#b0322b", "⛔ CIRCUIT BREAKER TRIPPED — recent portfolio "
                                 "stats breached limits; HOLD-only until probation reset"))
    if live_alpha_str:
        items.append(("#555", f"Live (weighted) alphas: <b>{live_alpha_str}</b>"
                              " — all others incubating in shadow mode"))
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


def table(sig_rows):
    if not sig_rows:
        return "<p style='color:#888'>None</p>"
    head = ("<tr style='background:#1a2b4a;color:#fff'>"
            "<th align='left'>Symbol</th><th>Price</th><th>Stop</th>"
            "<th>Target</th><th>R:R</th><th>Conf</th><th align='left'>Top alphas</th></tr>")
    body = ""
    for i, r in enumerate(sig_rows):
        bg = '#f4f7fb' if i % 2 else '#ffffff'
        # active alphas
        active = ""
        try:
            ab = ast.literal_eval(r.get('alpha_breakdown', '{}'))
            top = sorted(((k, v['score']) for k, v in ab.items()
                          if isinstance(v, dict) and v.get('score')),
                         key=lambda x: -abs(x[1]))[:3]
            active = ", ".join(f"{k} {v:+.2f}" for k, v in top)
        except Exception:
            pass
        body += (
            f"<tr style='background:{bg}'>"
            f"<td><b>{r.get('symbol','')}</b></td>"
            f"<td align='right'>{_fmt(r.get('price'))}</td>"
            f"<td align='right'>{_fmt(r.get('stop_loss'))}</td>"
            f"<td align='right'>{_fmt(r.get('target'))}</td>"
            f"<td align='center'>{_fmt(r.get('risk_reward'),1)}x</td>"
            f"<td align='center'>{_fmt(r.get('confidence'),0)}%</td>"
            f"<td style='font-size:12px;color:#555'>{active}</td>"
            f"</tr>"
        )
    return (f"<table cellpadding='6' cellspacing='0' "
            f"style='border-collapse:collapse;width:100%;font-size:14px'>"
            f"{head}{body}</table>")


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
        f"<h3 style='margin-bottom:4px'>Verified track record "
        f"<span style='font-weight:normal;font-size:12px;color:#888'>"
        f"(net of {cost:.2f}% costs, entry at next open, vs NIFTY)</span></h3>"
        f"{cum}"
        f"<table cellpadding='5' cellspacing='0' "
        f"style='border-collapse:collapse;font-size:13px'>"
        f"<tr style='background:#1a2b4a;color:#fff'><th>Window</th><th>Trades</th>"
        f"<th>Win rate</th><th>Avg net ret</th><th>Avg excess vs NIFTY</th></tr>"
        f"{rows}</table>"
    )


html = f"""<html><body style='font-family:Segoe UI,Arial,sans-serif;color:#222'>
<h2 style='margin-bottom:4px'>Daily Trading Signals — {today}</h2>
<p style='margin-top:0;color:#555'>
  Market regime: <b>{regime}</b>{f" &nbsp;|&nbsp; VIX: <b>{vix}</b>" if vix else ""}
  &nbsp;|&nbsp; {len(buys)} BUY / {len(sells)} SELL
  &nbsp;|&nbsp; recommended hold: <b>5 trading days</b>
</p>
{status_banner()}
<h3 style='color:#1a7a3a'>BUY ({len(buys)})</h3>
{table(buys)}
<h3 style='color:#b0322b'>SELL ({len(sells)})</h3>
{table(sells)}
{scoreboard()}
<p style='font-size:12px;color:#999;margin-top:20px'>
  Auto-generated by the multi-alpha engine on GitHub Actions after market
  close, using today's official closing data. Execute at the next market
  open. Use stop-losses.
</p>
</body></html>"""

Path('email_body.html').write_text(html, encoding='utf-8')
flag = ""
if breaker_tripped:
    flag = "[BREAKER] "
elif not gate_open:
    flag = "[GATE CLOSED] "
subject = f"{flag}Trading Signals {today}: {len(buys)} BUY / {len(sells)} SELL [{regime}]"
Path('email_subject.txt').write_text(subject, encoding='utf-8')
print("Wrote email_body.html and email_subject.txt")
print("Subject:", subject)
