"""
Cloud Engine Diagnostic
=======================
Runs ON GitHub Actions to prove whether the REAL multi_alpha_engine can run
in the cloud, and which data sources are reachable from GitHub's servers.

It writes a machine-readable report to:
    cloud_diagnostics/nse_access_report.json

The workflow commits that file back to the repo so we can read the result
locally via `git pull` (no gh CLI needed).

Tests:
  1. yfinance price fetch (the backbone of all price alphas)
  2. Each NSE / BSE endpoint the India alphas depend on (status + sample)
  3. NSE bhav-copy archive download (delivery_pct source)
  4. End-to-end run of the REAL engine on 3 stocks (quick mode), capturing
     the per-alpha breakdown so we see which alphas actually produce signal.
"""

import json
import sys
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path

OUT_DIR = Path("cloud_diagnostics")
OUT_DIR.mkdir(exist_ok=True)
REPORT = OUT_DIR / "nse_access_report.json"

report = {
    "run_at_utc": datetime.utcnow().isoformat(),
    "python": sys.version.split()[0],
    "tests": {},
}


def record(name, ok, detail):
    report["tests"][name] = {"ok": bool(ok), "detail": detail}
    flag = "PASS" if ok else "FAIL"
    print(f"[{flag}] {name}: {detail}")


# ---------------------------------------------------------------------------
# 1. yfinance
# ---------------------------------------------------------------------------
try:
    import yfinance as yf
    df = yf.download("RELIANCE.NS", period="1mo", progress=False, auto_adjust=False)
    if df is not None and len(df) > 5:
        record("yfinance", True, f"RELIANCE.NS -> {len(df)} rows, last close present")
    else:
        record("yfinance", False, f"empty/short frame ({0 if df is None else len(df)} rows)")
except Exception as e:
    record("yfinance", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 2. NSE / BSE JSON endpoints (the India-flow alphas)
# ---------------------------------------------------------------------------
import requests

_headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Accept": "application/json,text/html,*/*",
}

endpoints = {
    "nse_fiidii":   "https://www.nseindia.com/api/fiidiiTradeReact",
    "nse_bulk":     "https://www.nseindia.com/api/bulk-deals?optionType=bulk_deals",
    "nse_foban":    "https://nseindia.com/api/fo-ban-list",
    "bse_insider":  "https://api.bseindia.com/BseIndiaAPI/api/InsiderData/w?scripcode=&Flag=All",
}

for name, url in endpoints.items():
    try:
        r = requests.get(url, headers={**_headers, "Referer": "https://www.nseindia.com/"},
                         timeout=15)
        body = (r.text or "")[:160].replace("\n", " ")
        ok = r.status_code == 200 and len(r.text or "") > 50
        record(name, ok, f"HTTP {r.status_code}, {len(r.text or '')} bytes | {body!r}")
    except Exception as e:
        record(name, False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 3. NSE bhav-copy archive (delivery_pct source)
# ---------------------------------------------------------------------------
try:
    # Recent weekday
    d = date.today()
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    ymd = d.strftime("%Y%m%d")
    bhav_url = (f"https://nsearchives.nseindia.com/products/content/"
                f"sec_bhavdata_full_{ymd}.csv")
    r = requests.get(bhav_url, headers={**_headers, "Referer": "https://www.nseindia.com/"},
                     timeout=20)
    ok = r.status_code == 200 and len(r.text or "") > 500
    record("nse_bhav_archive", ok,
           f"HTTP {r.status_code}, {len(r.text or '')} bytes for {ymd}")
except Exception as e:
    record("nse_bhav_archive", False, f"{type(e).__name__}: {e}")


# ---------------------------------------------------------------------------
# 4. End-to-end REAL engine run (quick mode) on 3 stocks
# ---------------------------------------------------------------------------
test_syms = ["RELIANCE.NS", "TCS.NS", "ONGC.NS"]
try:
    import pandas as pd
    data_dir = Path("data/stocks")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Seed parquets via yfinance so the engine has price history to read.
    seeded = []
    for s in test_syms:
        try:
            d = yf.download(s, period="2y", progress=False, auto_adjust=False)
            if isinstance(d.columns, pd.MultiIndex):
                d.columns = d.columns.get_level_values(0)
            if d is not None and len(d) > 250:
                d.index = pd.to_datetime(d.index)
                keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
                        if c in d.columns]
                nse = s.replace(".NS", "").upper()
                d[keep].to_parquet(data_dir / f"{nse}.parquet")
                seeded.append(s)
        except Exception:
            pass
    record("engine_seed_data", len(seeded) == len(test_syms),
           f"seeded {len(seeded)}/{len(test_syms)} parquets")

    from multi_alpha_engine import MultiAlphaEngine
    engine = MultiAlphaEngine(data_dir=str(data_dir))
    ranked = engine.rank_universe(symbols=test_syms, as_of_date=date.today(),
                                  verbose=False)

    # Pull alpha breakdown for the top symbol to see which alphas fired.
    breakdown = {}
    if ranked is not None and len(ranked) > 0:
        top = ranked.iloc[0]
        comp = top.get("alpha_components", {}) if hasattr(top, "get") else {}
        if isinstance(comp, dict):
            for k, v in comp.items():
                sc = v.get("score") if isinstance(v, dict) else v
                breakdown[k] = sc
        record("engine_run", True,
               f"ranked {len(ranked)} stocks; top={top['symbol']} "
               f"score={top.get('composite_score')}")
    else:
        record("engine_run", False, "engine returned empty ranking")
    report["alpha_breakdown_top"] = breakdown
    # Which alphas are non-zero (actually contributing)
    active = [k for k, v in breakdown.items() if v not in (0, 0.0, None)]
    report["active_alphas"] = active
    print(f"[INFO] active alphas (top stock): {active}")

except Exception as e:
    record("engine_run", False, f"{type(e).__name__}: {e}")
    report["engine_traceback"] = traceback.format_exc()


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
t = report["tests"]
price_ok = t.get("yfinance", {}).get("ok") and t.get("engine_run", {}).get("ok")
nse_any = any(t.get(k, {}).get("ok") for k in
              ["nse_fiidii", "nse_bulk", "nse_foban", "bse_insider", "nse_bhav_archive"])
report["verdict"] = {
    "price_engine_works_in_cloud": bool(price_ok),
    "any_nse_source_reachable": bool(nse_any),
    "recommendation": (
        "Run real engine in cloud (price alphas + 5 improvements). "
        + ("NSE flow alphas also available." if nse_any else
           "NSE flow alphas dormant in cloud (same as local) — acceptable.")
    ) if price_ok else "Engine failed in cloud — investigate import/deps.",
}

REPORT.write_text(json.dumps(report, indent=2, default=str))
print("\n=== VERDICT ===")
print(json.dumps(report["verdict"], indent=2))
print(f"\nReport written to {REPORT}")
