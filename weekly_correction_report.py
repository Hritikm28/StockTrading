"""
Weekly Correction-Effectiveness Report
======================================
Answers the question: "Are the auto-corrections actually making us better?"

It reconstructs, from the outcome log (performance.csv), what the per-stock
dampener WOULD have decided at the moment each signal was generated, then
checks whether those decisions paid off:

  1. SYSTEM TREND     - win-rate & avg return per week (is the whole system
                        trending up?)
  2. DAMPENER CHECK   - bucket every trade by the dampener that was active
                        when it fired (suppressed / neutral / boosted) and
                        compare forward win-rates. If boosted > suppressed,
                        the correction mechanism is adding value.
  3. STOCK MOVERS     - which stocks improved / declined recently.

Writes:
  signals/weekly_correction_report_<date>.html   (committed + emailed)
  signals/weekly_correction_report_<date>.json
  correction_email_body.html  (transient, for the email step)

Pure stdlib + pandas/numpy. Degrades gracefully when data is thin.
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PERF = Path("paper_trading/results/performance.csv")
OUT_DIR = Path("signals")
OUT_DIR.mkdir(exist_ok=True)
today = date.today().strftime("%Y-%m-%d")

# Dampener config (must match multi_alpha_engine.PerformanceDampener)
LOOKBACK = 15
MIN_TRADES = 5
BASELINE = 0.55
MIN_MULT, MAX_MULT = 0.35, 1.15


def _load():
    if not PERF.exists():
        return pd.DataFrame()
    df = pd.read_csv(PERF)
    if "signal_date" not in df.columns or "symbol" not in df.columns:
        return pd.DataFrame()
    df["signal_date"] = pd.to_datetime(df["signal_date"], errors="coerce")
    df = df.dropna(subset=["signal_date"])
    # need at least 1-day outcome
    if "win_1d" not in df.columns and "ret_1d" in df.columns:
        df["win_1d"] = (df["ret_1d"] > 0).astype(float)
    return df.dropna(subset=["ret_1d"]).sort_values("signal_date")


def _dampener_at_signal(df):
    """For each trade, the dampener implied by the stock's PRIOR trades."""
    df = df.sort_values(["symbol", "signal_date"]).copy()
    mults = []
    for sym, grp in df.groupby("symbol"):
        wins = grp["win_1d"].tolist()
        for i in range(len(wins)):
            prior = [w for w in wins[max(0, i - LOOKBACK):i] if not pd.isna(w)]
            if len(prior) < MIN_TRADES:
                mults.append(1.0)
            else:
                wr = float(np.mean(prior))
                mults.append(float(np.clip(wr / BASELINE, MIN_MULT, MAX_MULT)))
    df["dampener"] = mults
    return df


def _bucket(m):
    if m < 0.9:
        return "suppressed"
    if m > 1.05:
        return "boosted"
    return "neutral"


def build():
    df = _load()
    report = {"date": today, "n_trades": int(len(df))}

    if len(df) < 10:
        report["status"] = "insufficient_data"
        report["message"] = f"Only {len(df)} graded trades — need ~10+ to assess."
        _write(report, None, None, None)
        return report

    # ── 1. System weekly trend ───────────────────────────────────────────────
    df["week"] = df["signal_date"].dt.to_period("W").dt.start_time.dt.date
    weekly = (df.groupby("week")
                .agg(n=("ret_1d", "size"),
                     win_rate=("win_1d", lambda s: round(100 * s.mean(), 1)),
                     avg_ret=("ret_1d", lambda s: round(s.mean(), 2)))
                .reset_index().tail(6))
    report["weekly"] = weekly.to_dict("records")

    # trend = slope of weekly win-rate (last up-to-4 weeks)
    wr = weekly["win_rate"].tail(4).tolist()
    if len(wr) >= 2:
        slope = (wr[-1] - wr[0]) / (len(wr) - 1)
        report["winrate_trend_per_week"] = round(slope, 2)

    # ── 2. Dampener effectiveness ────────────────────────────────────────────
    dd = _dampener_at_signal(df)
    dd["bucket"] = dd["dampener"].apply(_bucket)
    buck = (dd.groupby("bucket")
              .agg(n=("ret_1d", "size"),
                   win_rate=("win_1d", lambda s: round(100 * s.mean(), 1)),
                   avg_ret=("ret_1d", lambda s: round(s.mean(), 2)))
              .reindex(["boosted", "neutral", "suppressed"]).dropna(how="all")
              .reset_index())
    report["dampener_buckets"] = buck.to_dict("records")

    b = {r["bucket"]: r for r in report["dampener_buckets"]}
    works = None
    if "boosted" in b and "suppressed" in b:
        works = b["boosted"]["win_rate"] > b["suppressed"]["win_rate"]
    report["dampener_adds_value"] = works

    # ── 3. Stock movers: recent vs prior win-rate ────────────────────────────
    movers = []
    for sym, grp in df.groupby("symbol"):
        if len(grp) < 6:
            continue
        half = len(grp) // 2
        prior_wr = 100 * grp["win_1d"].iloc[:half].mean()
        recent_wr = 100 * grp["win_1d"].iloc[half:].mean()
        movers.append({"symbol": sym, "prior_wr": round(prior_wr, 0),
                       "recent_wr": round(recent_wr, 0),
                       "delta": round(recent_wr - prior_wr, 0),
                       "n": int(len(grp))})
    movers.sort(key=lambda x: x["delta"], reverse=True)
    report["improved"] = movers[:5]
    report["declined"] = [m for m in movers if m["delta"] < 0][-5:][::-1]

    # ── Verdict ──────────────────────────────────────────────────────────────
    verdict = []
    slope = report.get("winrate_trend_per_week")
    if slope is not None:
        verdict.append(("System win-rate trend",
                        f"{slope:+.1f}%/week",
                        "improving" if slope > 0 else ("flat" if slope == 0 else "declining")))
    if works is not None:
        verdict.append(("Dampener mechanism",
                        f"boosted {b['boosted']['win_rate']}% vs suppressed {b['suppressed']['win_rate']}%",
                        "adding value" if works else "not yet helping"))
    report["verdict"] = verdict

    _write(report, weekly, buck, (report["improved"], report["declined"]))
    return report


def _html(report, weekly, buck, movers):
    def tbl(df_or_rows, cols, headers):
        rows = df_or_rows.to_dict("records") if hasattr(df_or_rows, "to_dict") else df_or_rows
        if not rows:
            return "<p style='color:#888'>No data yet.</p>"
        th = "".join(f"<th>{h}</th>" for h in headers)
        body = ""
        for i, r in enumerate(rows):
            bg = "#f4f7fb" if i % 2 else "#fff"
            tds = "".join(f"<td align='center'>{r.get(c,'')}</td>" for c in cols)
            body += f"<tr style='background:{bg}'>{tds}</tr>"
        return (f"<table cellpadding='6' cellspacing='0' style='border-collapse:collapse;"
                f"width:100%;font-size:14px'><tr style='background:#1a2b4a;color:#fff'>"
                f"{th}</tr>{body}</table>")

    if report.get("status") == "insufficient_data":
        return (f"<html><body style='font-family:Segoe UI,Arial'>"
                f"<h2>Weekly Correction Report — {today}</h2>"
                f"<p>{report['message']}</p></body></html>")

    vparts = ""
    for name, val, tag in report.get("verdict", []):
        color = "#1a7a3a" if tag in ("improving", "adding value") else (
            "#b0322b" if tag in ("declining", "not yet helping") else "#888")
        vparts += (f"<li><b>{name}:</b> {val} "
                   f"<span style='color:{color};font-weight:bold'>({tag})</span></li>")

    return f"""<html><body style='font-family:Segoe UI,Arial,sans-serif;color:#222'>
<h2 style='margin-bottom:4px'>Weekly Correction Report — {today}</h2>
<p style='color:#555;margin-top:0'>Are the auto-corrections making us better?
({report['n_trades']} graded trades)</p>

<div style='background:#eef3fb;padding:10px 16px;border-radius:6px'>
<b>Verdict</b><ul style='margin:6px 0'>{vparts or '<li>Building baseline…</li>'}</ul>
</div>

<h3>1. System trend (by week)</h3>
{tbl(weekly, ['week','n','win_rate','avg_ret'], ['Week','Trades','Win %','Avg %'])}

<h3>2. Does the dampener add value?</h3>
<p style='color:#555;font-size:13px'>Trades grouped by the dampener active when they fired.
If <b>boosted</b> win-rate &gt; <b>suppressed</b>, the correction is working.</p>
{tbl(buck, ['bucket','n','win_rate','avg_ret'], ['Dampener','Trades','Win %','Avg %'])}

<h3>3. Stock movers (recent vs prior win-rate)</h3>
<b style='color:#1a7a3a'>Improved</b>
{tbl(movers[0] if movers else [], ['symbol','prior_wr','recent_wr','delta','n'], ['Stock','Prior %','Recent %','Δ','N'])}
<b style='color:#b0322b'>Declined</b>
{tbl(movers[1] if movers else [], ['symbol','prior_wr','recent_wr','delta','n'], ['Stock','Prior %','Recent %','Δ','N'])}

<p style='font-size:12px;color:#999;margin-top:18px'>
Auto-generated weekly. Reconstructs dampener decisions from the outcome log.</p>
</body></html>"""


def _write(report, weekly, buck, movers):
    (OUT_DIR / f"weekly_correction_report_{today}.json").write_text(
        json.dumps(report, indent=2, default=str))
    html = _html(report, weekly, buck, movers)
    (OUT_DIR / f"weekly_correction_report_{today}.html").write_text(html, encoding="utf-8")
    Path("correction_email_body.html").write_text(html, encoding="utf-8")
    print(f"Wrote weekly correction report for {today} ({report.get('n_trades',0)} trades)")


if __name__ == "__main__":
    r = build()
    for name, val, tag in r.get("verdict", []):
        print(f"  {name}: {val} ({tag})")
