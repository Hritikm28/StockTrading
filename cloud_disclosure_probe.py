"""
One-off probe: what do NSE's disclosure feed items link to, and what is in
them? Prints to the job log; commits nothing.
"""
import io
import re
import zipfile

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
BASE = "https://nsearchives.nseindia.com/content/RSS/"
FEEDS = ['InsiderTrading.xml', 'Sast_Regulation29.xml', 'Sast_Regulation31.xml',
         'Board_Meetings.xml', 'Financial_Results.xml']
s = requests.Session()
s.headers.update({'User-Agent': UA, 'Accept': '*/*'})


def show(label, data: bytes, n=2500):
    print(f"----- {label} ({len(data)} bytes)")
    print(data[:n].decode('utf-8', 'replace'))


for feed in FEEDS:
    print(f"\n================ {feed}")
    body = s.get(BASE + feed, timeout=20).text
    items = re.findall(r'<item>(.*?)</item>', body, re.S)
    print(f"items: {len(items)}")
    if items:
        show("first item raw", items[0].encode(), 800)
    for it in items[:2]:
        m = re.search(r'<link>(.*?)</link>', it, re.S)
        if not m:
            continue
        url = m.group(1).strip()
        try:
            r = s.get(url, timeout=30)
        except Exception as e:
            print(f"fetch failed {url}: {e}")
            continue
        print(f"link {url} -> HTTP {r.status_code}, {r.headers.get('content-type')}")
        data = r.content
        if data[:2] == b'PK':
            z = zipfile.ZipFile(io.BytesIO(data))
            print("zip contents:", z.namelist())
            for name in z.namelist()[:2]:
                show(f"zip member {name}", z.read(name), 2500)
        else:
            show("body", data, 3500)

for name, url in [('EQUITY_L', 'https://nsearchives.nseindia.com/content/equities/EQUITY_L.csv')]:
    try:
        r = s.get(url, timeout=20)
        print(f"\n================ {name}: HTTP {r.status_code}")
        print("\n".join(r.text.splitlines()[:4]))
    except Exception as e:
        print(f"{name} failed: {e}")
