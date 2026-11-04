"""
One-off probe: print complete insider-trading (PIT) XBRL filings from NSE's
RSS feed, with namespace declarations stripped, so the parser can be
written against real files. Prints to the job log; commits nothing.
"""
import re

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
s = requests.Session()
s.headers.update({'User-Agent': UA, 'Accept': '*/*'})
body = s.get("https://nsearchives.nseindia.com/content/RSS/InsiderTrading.xml",
             timeout=20).text
items = re.findall(r'<item>(.*?)</item>', body, re.S)
print(f"items: {len(items)}")
for it in items:
    print("ITEM", re.sub(r'\s+', ' ', it)[:400])
seen = set()
for it in items:
    link = re.search(r'<link>(.*?)</link>', it, re.S).group(1).strip()
    sym = re.search(r'<description>([^|]*)', it).group(1)
    if sym in seen or not link.endswith('.xml'):
        continue
    seen.add(sym)
    x = s.get(link, timeout=30).text
    x = re.sub(r'<xbrli:xbrl[^>]*>', '<xbrli:xbrl>', x)
    x = re.sub(r'<xbrli:context id="[^"]*">.*?</xbrli:context>', '', x, flags=re.S)
    print(f"\n======== {sym} {link}\n{x}")
    if len(seen) >= 3:
        break
