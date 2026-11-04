"""
Price-based factor definitions, shared by the live alphas and the research
lab (factor_lab.py) so a factor tested on history is exactly the one traded.

Every function takes a 1-D array of daily closes ending at the as-of date
(oldest first) and returns a float, or None when there is not enough clean
history. Returns are measured to one month ago: the latest month is skipped,
as in the academic 12-1 momentum definition, because one-month winners
tend to reverse.
"""

from typing import Optional

import numpy as np

YEAR = 252        # trading days
HALF = 126
SKIP = 21         # latest month left out of momentum windows


def _clean(c, n: int) -> Optional[np.ndarray]:
    if c is None or len(c) < n:
        return None
    w = np.asarray(c[-n:], dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        return None
    return w


def mom_12_1(c) -> Optional[float]:
    """12-month return to one month ago (the live momentum alpha)."""
    w = _clean(c, YEAR)
    return None if w is None else float(w[-SKIP - 1] / w[0] - 1)


def volatility(c) -> Optional[float]:
    """Annualised standard deviation of daily log returns over a year."""
    w = _clean(c, YEAR)
    if w is None:
        return None
    v = float(np.std(np.diff(np.log(w)), ddof=1) * np.sqrt(YEAR))
    return v if v > 0 else None


def riskadj_parts(c) -> Optional[tuple]:
    """(6-month return / vol, 12-month return / vol), both to one month ago.
    The NIFTY200 Momentum 30 index ranks on the average of these two after
    standardising each across the universe."""
    w = _clean(c, YEAR)
    vol = volatility(c)
    if w is None or vol is None:
        return None
    end = w[-SKIP - 1]
    return float((end / w[-HALF] - 1) / vol), float((end / w[0] - 1) / vol)


def discreteness(c) -> Optional[float]:
    """Information discreteness (Da, Gurun & Warachka 2014, "Frog in the
    pan"): sign(return) x (% down days - % up days) over the momentum
    window. Negative = the move came as many small steps (continuous),
    which persists; positive = a few big jumps, which fade."""
    w = _clean(c, YEAR)
    if w is None:
        return None
    r = np.diff(w[:-SKIP])
    ret = w[-SKIP - 1] / w[0] - 1
    return float(np.sign(ret) * ((r < 0).mean() - (r > 0).mean()))


def high_52w(c) -> Optional[float]:
    """Close as a fraction of its 52-week high (George & Hwang 2004)."""
    w = _clean(c, YEAR)
    return None if w is None else float(w[-1] / w.max())


def zscore(values: dict) -> dict:
    """Cross-sectional z-scores, winsorised at +/-3."""
    if len(values) < 3:
        return {}
    a = np.array(list(values.values()), dtype=float)
    sd = a.std()
    if sd == 0:
        return {k: 0.0 for k in values}
    z = np.clip((a - a.mean()) / sd, -3, 3)
    return dict(zip(values.keys(), z.tolist()))


def riskadj_scores(closes: dict) -> dict:
    """symbol -> combined z-score of risk-adjusted 6- and 12-month momentum."""
    parts = {k: riskadj_parts(c) for k, c in closes.items()}
    parts = {k: p for k, p in parts.items() if p is not None}
    z6 = zscore({k: p[0] for k, p in parts.items()})
    z12 = zscore({k: p[1] for k, p in parts.items()})
    return {k: (z6[k] + z12[k]) / 2 for k in z6}


def rank_to_score(values: dict) -> dict:
    """Map raw values to [-1, 1] by cross-sectional percentile rank."""
    if not values:
        return {}
    keys = sorted(values, key=values.get)
    n = len(keys)
    return {k: (i + 0.5) / n * 2 - 1 for i, k in enumerate(keys)}
