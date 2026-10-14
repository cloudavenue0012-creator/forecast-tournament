"""Walk-forward tournament: score every engine on each month that closes, and let
the scoreboard - not a person - decide which engine(s) forecast the next month.

Timeline (one step per month t):

  decision day  day ``asof_day`` of month t. Months 0..t-1 are closed; month t is
                open and only its days 1..k are visible.
  forecasts     every engine forecasts each horizon h = 0..3 (target t+h):
                  h0  the open month (nowcast) - "last month" in a close calendar,
                      where the month has ended but its books have not
                  h1  next month
                  h2, h3  two and three months out
                Each horizon is its own race with its own scoreboard.
  scoring       a forecast is scored only once its target month has closed, so
                at decision t the scoreboard holds targets <= t-1 only
  selection     per horizon, a recency-weighted MAPE over the latest
                ``selection_window`` scored targets ranks the engines. The leader
                runs, joined by any engine within ``band`` of it (at most
                ``top_k``), and the team's forecasts are averaged. When one engine
                is clearly ahead it runs alone; when several are statistically
                tied, averaging them beats betting on noise.

History comes from a store. ``ClosedStore`` serves booked values for closed
months only. ``MonthCache`` reproduces a real failure mode - a cache that assumes
"a past month never changes" and so freezes the month-to-date figure it saw
while the month was still open. Two gates audit every forecast:

  final-history gate  every month an engine reads, and every actual a score is
                      computed against, must be a closed (final) value
  event-window gate   no engine may read a promotion month as baseline history
                      (a rolling window that does turns a one-off into a "trend")
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engines import ENGINES, Info, forecast
from synth_series import KIND, Dataset

HORIZONS = (0, 1, 2, 3)
HORIZON_LABEL = {0: "h0 open month (nowcast)", 1: "h1 next month",
                 2: "h2 +2 months", 3: "h3 +3 months"}

# The shipped selection rule. Chosen on tuning seeds 100-119 only; every number in
# the README is measured on other seeds (42 and 0-49), so the rule is not fitted
# to the results it is reported on.
RULE = dict(selection_window=24, half_life=6.0, band=0.20, top_k=3)


class GateViolation(RuntimeError):
    pass


# -- history stores ----------------------------------------------------------------

class ClosedStore:
    """Booked values for closed months; nothing for the open month."""

    def __init__(self, ds: Dataset, metric: str):
        self.values = ds.values(metric)

    def observe(self, t: int, mtd_value: float) -> None:
        pass

    def history(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        return self.values[:t].copy(), np.ones(t, dtype=bool)


class MonthCache:
    """A monthly cache keyed by month. Each decision day it records the open
    month's month-to-date figure (that is what a dashboard shows mid-month).

    ``refresh_unclosed=False`` is the bug: the cache treats every cached month
    as immutable, so once month t is cached mid-month its partial figure is
    never replaced by the booked value after close. ``refresh_unclosed=True``
    tags each entry final / not-final and refetches non-final ones.
    """

    def __init__(self, ds: Dataset, metric: str, refresh_unclosed: bool):
        self.booked = ds.values(metric)
        self.refresh = refresh_unclosed
        self.cache: dict[int, tuple[float, bool]] = {}

    def observe(self, t: int, mtd_value: float) -> None:
        for m in range(t):
            hit = self.cache.get(m)
            if hit is None or (self.refresh and not hit[1]):
                self.cache[m] = (float(self.booked[m]), True)
        if t not in self.cache or self.refresh:
            self.cache[t] = (mtd_value, False)

    def history(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        vals = np.array([self.cache[m][0] for m in range(t)])
        final = np.array([self.cache[m][1] for m in range(t)], dtype=bool)
        return vals, final


# -- gates -------------------------------------------------------------------------

def final_history_violations(used: tuple[int, ...], final: np.ndarray) -> list[int]:
    return [m for m in used if not final[m]]


def event_window_violations(used: tuple[int, ...], events: np.ndarray) -> list[int]:
    return [m for m in used if events[m]]


# -- the run -----------------------------------------------------------------------

@dataclass
class Result:
    metric: str
    forecasts: pd.DataFrame        # one row per (decision t, horizon, engine)
    selected: pd.DataFrame         # one row per (decision t, horizon): team + its forecast
    scored_actuals: dict           # target -> the actual the run scored against
    gate_final: int = 0            # months read / scored that were not final
    gate_event: int = 0            # promotion months read as baseline history


def leaderboard(errs: dict[int, dict[str, float]], names: list[str], window: int,
                half_life: float | None) -> dict[str, float]:
    """Recency-weighted MAPE per engine over the latest ``window`` scored targets."""
    scored = sorted(errs)[-window:]
    w = (np.ones(len(scored)) if half_life is None
         else 0.5 ** ((len(scored) - 1 - np.arange(len(scored))) / half_life))
    return {e: float(np.average([errs[m][e] for m in scored], weights=w)) for e in names}


def pick_team(board: dict[str, float], names: list[str], rule: str, band: float,
              top_k: int) -> list[str]:
    ranked = sorted(names, key=lambda e: (board[e], names.index(e)))
    if rule == "champion":
        return ranked[:1]
    return [e for e in ranked if board[e] <= board[ranked[0]] * (1 + band)][:top_k]


def run(ds: Dataset, metric: str, *, selection_window: int = RULE["selection_window"],
        half_life: float | None = RULE["half_life"], band: float = RULE["band"],
        top_k: int = RULE["top_k"], rule: str = "band", min_scored: int = 3,
        default: str = "mean3", mask_events: bool = True, store=None, strict: bool = True,
        first_decision: int = 14) -> Result:
    store = store or ClosedStore(ds, metric)
    events, days, promo_days = ds.events(), ds.days(), ds.promo_days()
    n, k = ds.n, ds.cfg.asof_day
    mtd_raw = ds.mtd_series(metric, k)                 # what a mid-month dashboard shows
    mtd_base = ds.mtd_series(metric, k, baseline=True)  # promotion days removed (nowcast input)
    names = list(ENGINES)

    rows, sel, scored_actuals = [], [], {}
    by_target: dict[int, list[dict]] = {}
    errors: dict[int, dict[int, dict[str, float]]] = {h: {} for h in HORIZONS}
    gate_final = gate_event = 0

    for t in range(first_decision, n):
        store.observe(t, mtd_raw[t])
        hist, final = store.history(t)

        # 1) month t-1 has closed: score every forecast that targeted it
        if t - 1 in by_target:
            actual = hist[t - 1]
            scored_actuals[t - 1] = actual
            if not final[t - 1]:
                gate_final += 1
                if strict:
                    raise GateViolation(f"scoring {metric} month {t-1} against a non-final value")
            for r in by_target.pop(t - 1):
                errors[r["h"]].setdefault(t - 1, {})[r["engine"]] = abs(r["forecast"] - actual) / abs(actual)

        info = Info(t=t, kind=KIND[metric], history=hist, mtd=float(mtd_base[t]),
                    mtd_hist=mtd_base[:t], events=events, days=days, promo_days=promo_days,
                    asof_day=k, mask_events=mask_events)

        for h in HORIZONS:
            target = t + h
            if target >= n:
                continue
            # 2) team from this horizon's scoreboard (closed targets only)
            if len(errors[h]) >= min_scored:
                board = leaderboard(errors[h], names, selection_window, half_life)
                team = pick_team(board, names, rule, band, top_k)
            else:
                team = [default]
            # 3) every engine forecasts; the gates audit what each one read
            out = {}
            for e in names:
                f = forecast(e, info, target)
                bad_final = final_history_violations(f.used + f.overlay_used, final)
                bad_event = event_window_violations(f.used, events)
                gate_final += len(bad_final)
                gate_event += len(bad_event)
                if strict and bad_final:
                    raise GateViolation(f"{e} read non-final {metric} months {bad_final} at decision {t}")
                if strict and bad_event:
                    raise GateViolation(f"{e} read promotion months {bad_event} as baseline at decision {t}")
                row = dict(t=t, h=h, target=target, engine=e, forecast=f.value)
                rows.append(row)
                by_target.setdefault(target, []).append(row)
                out[e] = f.value
            sel.append(dict(t=t, h=h, target=target, team="+".join(team),
                            forecast=float(np.mean([out[e] for e in team]))))

    return Result(metric, pd.DataFrame(rows), pd.DataFrame(sel), scored_actuals,
                  gate_final, gate_event)


# -- scoring against the truth ------------------------------------------------------

EVAL_FROM = 36   # targets 36..83 (four years, straddling the regime break) are scored


def ape(forecast: pd.Series, actual: np.ndarray) -> pd.Series:
    return (forecast - actual).abs() / np.abs(actual)


def score(ds: Dataset, res: Result, eval_from: int = EVAL_FROM) -> pd.DataFrame:
    """MAPE per horizon for the tournament, every fixed engine, the default engine,
    a 'frozen' pick and the ex-post oracle - always against booked ground truth,
    never against whatever a store served.

    frozen   the engine with the best MAPE on the warm-up targets that were already
             closed at the first evaluated decision, then kept forever - what a
             careful analyst does once, by hand
    oracle   the best engine for each target month, chosen after the fact (a floor
             no selection rule can reach)
    """
    truth = ds.values(res.metric)
    fc = res.forecasts.copy()
    fc["ape"] = ape(fc.forecast, truth[fc.target])
    sel = res.selected.copy()
    sel["ape"] = ape(sel.forecast, truth[sel.target])

    out = []
    for h in HORIZONS:
        f = fc[fc.h == h]
        live, warm = f[f.target >= eval_from], f[f.target < eval_from - h]
        fixed = live.groupby("engine").ape.mean()
        frozen = warm.groupby("engine").ape.mean().idxmin()
        s = sel[(sel.h == h) & (sel.target >= eval_from)]
        row = {"metric": res.metric, "h": h, "n": int(len(s)),
               "tournament": float(s.ape.mean()),
               "default": float(fixed["mean3"]),
               "frozen": frozen, "frozen_mape": float(fixed[frozen]),
               "best_fixed": fixed.idxmin(), "best_fixed_mape": float(fixed.min()),
               "oracle": float(live.groupby("target").ape.min().mean())}
        row.update({f"mape_{e}": float(fixed[e]) for e in ENGINES})
        out.append(row)
    return pd.DataFrame(out)
