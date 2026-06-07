"""
NSE/BSE URL probe — runs on GitHub Actions to find which data endpoints work
from GitHub's servers, using CORRECTED URLs + a cookie-session handshake.

Why: the first diagnostic 404'd on bulk-deals / fo-ban / bhav because those
used stale URLs and/or the wrong date format, and NSE's JSON APIs need a
cookie handshake (hit the homepage first). NSE's *archive* CSV endpoints
(nsearchives.nseindia.com) are static and usually need no cookies.

Writes cloud_diagnostics/nse_url_probe.json (committed back by the workflow).
"""
import json
from datetime import date, timedelta
from pathlib import Path

import requests

OUT = Path("cloud_diagnostics")
OUT.mkdir(exist_ok=True)

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
BASE_HEADERS = {"User-Agent": UA, "Accept-Language": "en-US,en;q=0.9"}

results = {}


def rec(name, url, ok, detail):
    results[name] = {"url": url, "ok": bool(ok), "detail": detail}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def nse_session():
    """Session primed with NSE cookies (needed for /api/ JSON endpoints)."""
    s = requests.Session()
    s.headers.update(BASE_HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=15)
        s.get("https://www.nseindia.com/market-data/securities-available-for-trading",
              timeout=15)
    except Exception:
        pass
    return s


# most recent weekday
d = date.today()
while d.weekday() >= 5:
    d -= timedelta(days=1)
ddmmyyyy = d.strftime("%d%m%Y")
yyyymmdd = d.strftime("%Y%m%d")

# ── Static archive CSVs (no cookies expected) ────────────────────────────────
archive_csvs = {
    "foban_csv":        "https://nsearchives.nseindia.com/content/fo/fo_secban.csv",
    "bulk_csv":         "https://nsearchives.nseindia.com/content/equities/bulk.csv",
    "block_csv":        "https://nsearchives.nseindia.com/content/equities/block.csv",
    "bhav_full_ddmmyyyy": f"https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{ddmmyyyy}.csv",
    "bhav_udiff_zip":   f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{yyyymmdd}_F_0000.csv.zip",
}
for name, url in archive_csvs.items():
    try:
        r = requests.get(url, headers={**BASE_HEADERS, "Referer": "https://www.nseindia.com/"},
                         timeout=25)
        body = (r.text or "")[:120].replace("\n", " ") if "zip" not in name else f"{len(r.content)} bytes"
        ok = r.status_code == 200 and (len(r.content) > 300)
        rec(name, url, ok, f"HTTP {r.status_code} | {body!r}")
    except Exception as e:
        rec(name, url, False, f"{type(e).__name__}: {e}")

# ── JSON APIs (need cookie session) ──────────────────────────────────────────
s = nse_session()
from_d = (d - timedelta(days=30)).strftime("%d-%m-%Y")
to_d = d.strftime("%d-%m-%Y")
json_apis = {
    "fiidii":            "https://www.nseindia.com/api/fiidiiTradeReact",
    "bulk_hist_json":    f"https://www.nseindia.com/api/historical/bulk-deals?from={from_d}&to={to_d}",
    "foban_json":        "https://www.nseindia.com/api/snapshot-derivatives-equity?index=ban_list",
}
for name, url in json_apis.items():
    try:
        r = s.get(url, headers={"Referer": "https://www.nseindia.com/"}, timeout=20)
        body = (r.text or "")[:120].replace("\n", " ")
        ok = r.status_code == 200 and r.text.strip().startswith(("[", "{"))
        rec(name, url, ok, f"HTTP {r.status_code} | {body!r}")
    except Exception as e:
        rec(name, url, False, f"{type(e).__name__}: {e}")

# ── BSE insider candidates ───────────────────────────────────────────────────
bse = {
    "bse_insider_new": "https://api.bseindia.com/BseIndiaAPI/api/InsiderTrading_New/w?scripcode=&Flag=All",
}
for name, url in bse.items():
    try:
        r = requests.get(url, headers={**BASE_HEADERS, "Referer": "https://www.bseindia.com/"},
                         timeout=20)
        body = (r.text or "")[:120].replace("\n", " ")
        ok = r.status_code == 200 and r.text.strip().startswith(("[", "{"))
        rec(name, url, ok, f"HTTP {r.status_code} | {body!r}")
    except Exception as e:
        rec(name, url, False, f"{type(e).__name__}: {e}")

summary = {
    "probe_date": str(d),
    "working": sorted([k for k, v in results.items() if v["ok"]]),
    "broken": sorted([k for k, v in results.items() if not v["ok"]]),
    "results": results,
}
(OUT / "nse_url_probe.json").write_text(json.dumps(summary, indent=2))
print("\nWORKING:", summary["working"])
print("BROKEN :", summary["broken"])
