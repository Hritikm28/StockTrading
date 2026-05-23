"""
Publish signals from daily_runner.py output into the signals/ directory.
Called by GitHub Actions after running daily_runner.py --quick.
Normalises column names so weekly_review.py always finds 'price'.
"""
import pandas as pd
import json
import shutil
from pathlib import Path
from datetime import date

today     = date.today().strftime('%Y-%m-%d')
dst       = Path('signals')
dst.mkdir(exist_ok=True)

# ── Copy CSV ──────────────────────────────────────────────────────────────────
src_csv = Path('paper_trading') / 'records' / f'signals_{today}.csv'
if src_csv.exists():
    df = pd.read_csv(src_csv)
    if 'current_price' in df.columns and 'price' not in df.columns:
        df = df.rename(columns={'current_price': 'price'})
    if 'date' not in df.columns:
        df.insert(0, 'date', today)
    df.to_csv(dst / f'{today}_approved.csv', index=False)
    df.to_csv(dst / 'latest.csv', index=False)
    print(f"Published {len(df)} approved signals to signals/{today}_approved.csv")
else:
    print(f"No approved signals today ({src_csv} not found). Writing empty latest.csv.")
    pd.DataFrame(columns=[
        'date', 'symbol', 'signal', 'composite_score',
        'confidence', 'price', 'stop_loss', 'target',
        'risk_reward', 'regime'
    ]).to_csv(dst / 'latest.csv', index=False)

# ── Copy JSON ─────────────────────────────────────────────────────────────────
src_json = Path('daily_reports') / f'report_{today}.json'
if src_json.exists():
    shutil.copy(src_json, dst / f'{today}_summary.json')
    print(f"Published report to signals/{today}_summary.json")
