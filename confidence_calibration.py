"""
Per-Alpha Confidence Calibration
================================
The alpha modules report hardcoded confidences (momentum is always 70,
mean_rev 65/80...) — numbers that were never checked against outcomes.
This module makes confidence EARNED: it joins the per-trade alpha
snapshots (paper_trading/alpha_scores/*.json) with graded outcomes
(paper_trading/results/performance.csv) and measures, for each alpha,
how often the trades it agreed with actually won.

Output: paper_trading/alpha_confidence.json
    { "<alpha>": {"multiplier": 0.82, "win_rate": 45.1, "n": 37}, ... }

multiplier = win_rate / 55%  (55% ≈ break-even after 0.4% costs),
clipped to [0.50, 1.15], only applied once n >= MIN_N trades of
evidence exist; otherwise the alpha keeps multiplier 1.0 (neutral).

multi_alpha_engine scales each alpha's reported confidence by its
multiplier, so persistently wrong alphas lose their say in the
confidence gate without being hard-killed.

Usage (runs in the daily workflow right after grading):
    python confidence_calibration.py
"""

import json
from pathlib import Path

import pandas as pd

ALPHA_DIR   = Path("paper_trading/alpha_scores")
PERF_CSV    = Path("paper_trading/results/performance.csv")
OUT_FILE    = Path("paper_trading/alpha_confidence.json")

MIN_N       = 15      # trades of evidence before calibration kicks in
BREAKEVEN   = 55.0    # % win rate ≈ breakeven after costs
MIN_MULT    = 0.50
MAX_MULT    = 1.15
AGREE_BAR   = 0.2     # |score| needed to count as "the alpha agreed"


def compute() -> dict:
    if not PERF_CSV.exists():
        return {}
    perf = pd.read_csv(PERF_CSV)
    if 'win_5d' not in perf.columns or perf.empty:
        return {}
    perf = perf.dropna(subset=['win_5d'])

    stats: dict = {}
    for _, row in perf.iterrows():
        symbol = str(row['symbol']).replace('.NS', '')
        f = ALPHA_DIR / f"{symbol}_{row['signal_date']}.json"
        if not f.exists():
            continue
        try:
            comps = json.loads(f.read_text())
        except Exception:
            continue
        direction = 1.0 if str(row.get('signal', 'BUY')).upper() == 'BUY' else -1.0
        won = bool(row['win_5d'])
        for name, comp in comps.items():
            score = float(comp.get('score', 0.0) or 0.0)
            if score * direction > AGREE_BAR:   # alpha backed this trade
                s = stats.setdefault(name, {'n': 0, 'wins': 0})
                s['n'] += 1
                s['wins'] += int(won)

    out = {}
    for name, s in stats.items():
        n = s['n']
        wr = 100.0 * s['wins'] / n if n else 0.0
        if n >= MIN_N:
            mult = max(MIN_MULT, min(MAX_MULT, wr / BREAKEVEN))
        else:
            mult = 1.0
        out[name] = {'multiplier': round(mult, 3),
                     'win_rate': round(wr, 1), 'n': n}
    return out


def main():
    out = compute()
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(out, indent=2))
    print(f"Calibrated {len(out)} alphas -> {OUT_FILE}")
    for name, s in sorted(out.items(), key=lambda kv: kv[1]['multiplier']):
        flag = '' if s['n'] >= MIN_N else '  (neutral: thin evidence)'
        print(f"  {name:<14} mult {s['multiplier']:.2f}  "
              f"wr {s['win_rate']:5.1f}%  n={s['n']}{flag}")


if __name__ == '__main__':
    main()
