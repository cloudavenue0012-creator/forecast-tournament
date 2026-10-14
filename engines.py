"""The competing forecast engines.

Every engine is a pure function ``engine(info, target) -> Forecast``. ``Info`` is
the *only* thing an engine can see, and it is built to contain nothing that
would not be known on the decision day:

  history    closed (booked) values for months 0..t-1 - month t is still open
  mtd        month-to-date operational value of the open month t, days 1..k,
             with any promotion days already run taken out (baseline)
  mtd_hist   the same day-k baseline month-to-date value for each closed month,
             so a nowcast can learn how far day-k figures sit from the close
  events     the promotion calendar - planned, therefore known in advance,
             including how many days each promotion runs
  days       calendar days per month

Each forecast also returns ``used``: the closed months it read. That provenance
is what the gates in ``tournament.py`` audit - an engine cannot quietly read an
unclosed month or a promotion month without it showing up.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class Info:
    t: int                     # decision month: open, not yet closed
    kind: str                  # "additive" (ratio) | "multiplicative" (amount)
    history: np.ndarray        # closed values, months 0..t-1
    mtd: float                 # open month t, days 1..asof_day, promotion days removed
    mtd_hist: np.ndarray       # the same baseline MTD for months 0..t-1
    events: np.ndarray         # planned promotion calendar (full horizon)
    days: np.ndarray           # calendar days per month (full horizon)
    promo_days: np.ndarray     # planned promotion days per month (full horizon)
    asof_day: int
    mask_events: bool = True   # drop promotion months from rolling windows


@dataclass(frozen=True)
class Forecast:
    value: float
    used: tuple[int, ...]      # closed months read as baseline history (audit trail)
    overlay_used: tuple[int, ...] = ()   # closed months read by the promotion overlay


def _clean(info: Info, i: int) -> bool:
    return 0 <= i < info.t and not (info.mask_events and info.events[i])


def window(info: Info, n: int, lookback: int = 36) -> list[int]:
    """Indices of the latest ``n`` usable closed months (newest first)."""
    out = []
    for i in range(info.t - 1, max(info.t - 1 - lookback, -1), -1):
        if _clean(info, i):
            out.append(i)
            if len(out) == n:
                break
    return out


def _mean(info: Info, idx: list[int]) -> Forecast:
    return Forecast(float(np.mean(info.history[idx])), tuple(idx))


# -- the engines -----------------------------------------------------------------

def last(info: Info, target: int) -> Forecast:
    """Previous closed month."""
    return _mean(info, window(info, 1))


def mean3(info: Info, target: int) -> Forecast:
    """3-month rolling average."""
    return _mean(info, window(info, 3))


def mean12(info: Info, target: int) -> Forecast:
    """12-month rolling average."""
    return _mean(info, window(info, 12))


def seasonal(info: Info, target: int) -> Forecast:
    """Same month last year (falls back two years, then to mean3, if that month was a promotion)."""
    for back in (12, 24):
        if _clean(info, target - back):
            return Forecast(float(info.history[target - back]), (target - back,))
    return mean3(info, target)


def seasonal_growth(info: Info, target: int) -> Forecast:
    """Same month last year, moved by the year-on-year change of the latest 3 clean month pairs."""
    base = seasonal(info, target)
    pairs = [(i, i - 12) for i in window(info, 12) if _clean(info, i - 12)][:3]
    if not pairs:
        return base
    now = info.history[[p[0] for p in pairs]]
    ago = info.history[[p[1] for p in pairs]]
    used = tuple(sorted(set(base.used) | {j for p in pairs for j in p}))
    if info.kind == "additive":
        return Forecast(base.value + float(now.mean() - ago.mean()), used)
    return Forecast(base.value * float(now.sum() / ago.sum()), used)


def ses(info: Info, target: int, alpha: float = 0.4) -> Forecast:
    """Simple exponential smoothing over the last 24 clean months."""
    idx = sorted(window(info, 24))
    level = info.history[idx[0]]
    for i in idx[1:]:
        level = alpha * info.history[i] + (1 - alpha) * level
    return Forecast(float(level), tuple(idx))


def trend(info: Info, target: int) -> Forecast:
    """Linear regression on the last 12 clean months, extrapolated to the target."""
    idx = sorted(window(info, 12))
    if len(idx) < 3:
        return _mean(info, idx)
    slope, icept = np.polyfit(idx, info.history[idx], 1)
    return Forecast(float(icept + slope * target), tuple(idx))


def _nowcast_open_month(info: Info) -> Forecast:
    """Open month from its day-k figure, corrected by the day-k -> close gap seen in
    the last 12 clean closed months (additive for ratios, a run-rate factor for amounts)."""
    idx = window(info, 12)
    k, t = info.asof_day, info.t
    if info.kind == "additive":
        gap = float(np.mean(info.history[idx] - info.mtd_hist[idx]))
        return Forecast(info.mtd + gap, tuple(idx))
    runrate = info.mtd_hist[idx] * info.days[idx] / k
    factor = float(np.mean(info.history[idx] / runrate))
    return Forecast(info.mtd * info.days[t] / k * factor, tuple(idx))


def nowcast(info: Info, target: int) -> Forecast:
    """h=0: the open month's baseline from its own partial data (non-promotion days).
    h>=1: a 3-month average whose newest point is that nowcast."""
    if np.isnan(info.mtd):            # every observed day was a promotion day
        return mean3(info, target)
    nc = _nowcast_open_month(info)
    if target == info.t:
        return nc
    idx = window(info, 2)
    vals = [nc.value] + [float(v) for v in info.history[idx]]
    return Forecast(float(np.mean(vals)), tuple(sorted(set(nc.used) | set(idx))))


ENGINES: dict[str, Callable[[Info, int], Forecast]] = {
    "last": last,
    "mean3": mean3,
    "mean12": mean12,
    "seasonal": seasonal,
    "seasonal_growth": seasonal_growth,
    "ses": ses,
    "trend": trend,
    "nowcast": nowcast,
}

# engines whose window rolls forward month by month - the ones a one-off
# promotion can leak into (the trap that event masking exists to stop)
ROLLING = ("last", "mean3", "mean12", "ses", "trend")


# -- the shared promotion overlay -------------------------------------------------

def promo_effect(info: Info, max_events: int = 6) -> tuple[float | None, tuple[int, ...]]:
    """Effect of one promotion *day-share*, learned from closed promotion months only:
    each closed promotion month is compared with the mean of the (up to) 3 clean months
    before it, and the gap is divided by the share of the month the promotion ran."""
    effects, used = [], set()
    for j in range(info.t):
        if not info.events[j]:
            continue
        base_idx = [i for i in range(j - 1, max(j - 7, -1), -1) if not info.events[i]][:3]
        if not base_idx:
            continue
        base = float(np.mean(info.history[base_idx]))
        share = info.promo_days[j] / info.days[j]
        gap = info.history[j] - base if info.kind == "additive" else info.history[j] / base - 1
        effects.append(gap / share)
        used.update(base_idx + [j])
    if not effects:
        return None, ()
    return float(np.mean(effects[-max_events:])), tuple(sorted(used))


def forecast(name: str, info: Info, target: int) -> Forecast:
    """Run one engine, then - if the target is a planned promotion month - apply the
    shared overlay. The overlay is identical for every engine, so it cannot tilt the
    tournament (the h=0 nowcast reads only the non-promotion days of the open month,
    so it too forecasts a baseline and takes the same overlay)."""
    f = ENGINES[name](info, target)
    if not info.events[target]:
        return f
    effect, used = promo_effect(info)
    if effect is None:
        return f
    share = info.promo_days[target] / info.days[target]
    value = f.value + effect * share if info.kind == "additive" else f.value * (1 + effect * share)
    return Forecast(value, f.used, used)
