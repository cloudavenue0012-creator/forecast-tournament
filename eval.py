"""Evaluation harness - the differentiator.

Three questions, each measured against the planted ground truth:

  1. selection   does letting the scoreboard pick beat the alternatives a person
                 would actually run - the default engine (3-month average), or the
                 engine that looked best on the warm-up period, frozen forever?
                 How far is it from the best fixed engine in hindsight, and from
                 the per-month oracle?
  2. robustness  the same comparison on 50 held-out seeds (the selection rule was
                 tuned on seeds 100-119, none of which are reported here)
  3. traps       two failure modes reproduced on purpose, with the gate that must
                 catch each:
                   A  a rolling window that swallows a one-off promotion month and
                      carries it forward as if it were the trend
                   B  a month cache that freezes the month-to-date figure it saw
                      while the month was open, so a partial month becomes history

Exit code 1 if any requested gate fails (see the flags), so CI goes red.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from engines import ENGINES
from synth_series import METRICS, generate
from tournament import (EVAL_FROM, HORIZON_LABEL, HORIZONS, ClosedStore, GateViolation,
                        MonthCache, ape, run, score)

ROLLING_SHOWN = ("mean3", "trend")
RUN_KW: dict = {}   # selection-rule overrides from the command line (default: the shipped RULE)


# -- 1 + 2: selection ---------------------------------------------------------------

def seed_scores(seed: int) -> pd.DataFrame:
    ds = generate(seed)
    return pd.concat([score(ds, run(ds, m, **RUN_KW)) for m in METRICS], ignore_index=True)


def holdout(seeds: range) -> pd.DataFrame:
    d = pd.concat([seed_scores(s).assign(seed=s) for s in seeds], ignore_index=True)
    d["beat_default"] = d.tournament < d.default
    d["beat_frozen"] = d.tournament <= d.frozen_mape
    d["beat_best"] = d.tournament <= d.best_fixed_mape
    d["regret"] = d.tournament / d.best_fixed_mape - 1
    d["gap_closed"] = (d.default - d.tournament) / (d.default - d.oracle)
    return d


def regime_cost(seeds: range, h: int = 1) -> pd.DataFrame:
    """Where the tournament loses to the hindsight-best fixed engine: MAPE by period
    around the regime break (h1), averaged over seeds."""
    rows = []
    for seed in seeds:
        ds = generate(seed)
        brk = ds.cfg.regime_month
        periods = {"before break": (EVAL_FROM, brk), "first 12m after": (brk, brk + 12),
                   "later": (brk + 12, ds.n)}
        for metric in METRICS:
            truth = ds.values(metric)
            res = run(ds, metric, **RUN_KW)
            f = res.forecasts[(res.forecasts.h == h) & (res.forecasts.target >= EVAL_FROM)].copy()
            f["ape"] = ape(f.forecast, truth[f.target])
            best = f.groupby("engine").ape.mean().idxmin()
            s = res.selected[(res.selected.h == h) & (res.selected.target >= EVAL_FROM)].copy()
            s["ape"] = ape(s.forecast, truth[s.target])
            for name, (lo, hi) in periods.items():
                rows.append(dict(metric=metric, period=name,
                                 tournament=s[(s.target >= lo) & (s.target < hi)].ape.mean(),
                                 best_fixed=f[(f.engine == best) & (f.target >= lo) & (f.target < hi)].ape.mean()))
    return pd.DataFrame(rows).groupby(["metric", "period"], sort=False).mean()


# -- 3A: promotion leaking into rolling windows -------------------------------------

def shadow_targets(ds, h: int) -> np.ndarray:
    """Non-promotion target months whose decision month had a promotion among the
    3 months just closed - the months a leaking rolling window gets wrong."""
    ev = ds.events()
    out = []
    for target in range(EVAL_FROM, ds.n):
        t = target - h
        if not ev[target] and ev[max(t - 3, 0):t].any():
            out.append(target)
    return np.array(out)


def promotion_trap(seed: int, h: int = 1) -> dict:
    ds = generate(seed)
    out = {}
    for metric in METRICS:
        truth = ds.values(metric)
        shadow = shadow_targets(ds, h)
        masked = run(ds, metric, mask_events=True)
        leaky = run(ds, metric, mask_events=False, strict=False)
        try:
            run(ds, metric, mask_events=False, strict=True)
            caught = False
        except GateViolation:
            caught = True
        rows = {}
        for label, res in (("masked", masked), ("leaky", leaky)):
            f = res.forecasts[(res.forecasts.h == h) & res.forecasts.target.isin(shadow)]
            for e in ROLLING_SHOWN:
                g = f[f.engine == e]
                err = g.forecast.to_numpy() - truth[g.target]
                rows[(e, label)] = dict(mape=float(ape(g.forecast, truth[g.target]).mean()),
                                        bias=float(np.mean(err / truth[g.target])))
            s = res.selected[(res.selected.h == h) & res.selected.target.isin(shadow)]
            rows[("tournament", label)] = dict(mape=float(ape(s.forecast, truth[s.target]).mean()),
                                               bias=float(np.mean((s.forecast - truth[s.target]) / truth[s.target])))
        out[metric] = dict(n_shadow=len(shadow), n_events=int(ds.events().sum()),
                           violations=leaky.gate_event, caught=caught,
                           violations_masked=masked.gate_event, rows=rows)
    return out


# -- 3B: the frozen month cache -----------------------------------------------------

def cache_trap(seed: int) -> dict:
    ds = generate(seed)
    out = {}
    for metric in METRICS:
        truth = ds.values(metric)
        clean = run(ds, metric)
        cache = MonthCache(ds, metric, refresh_unclosed=False)
        buggy = run(ds, metric, store=cache, strict=False)
        frozen_months = sum(1 for m, (_, fin) in cache.cache.items() if not fin and m < ds.n - 1)
        try:
            run(ds, metric, store=MonthCache(ds, metric, refresh_unclosed=False), strict=True)
            caught = False
        except GateViolation:
            caught = True
        fixed = run(ds, metric, store=MonthCache(ds, metric, refresh_unclosed=True), strict=True)

        def mape(res, actual_of):
            s = res.selected[res.selected.target >= EVAL_FROM]
            act = np.array([actual_of(m) for m in s.target])
            return float(ape(s.forecast, act).mean())

        reported = mape(buggy, lambda m: buggy.scored_actuals.get(m, truth[m]))
        out[metric] = dict(
            violations=buggy.gate_final, frozen_months=frozen_months, caught=caught, violations_fixed=fixed.gate_final,
            fixed_identical=bool(np.allclose(fixed.selected.forecast, clean.selected.forecast)),
            true_clean=mape(clean, lambda m: truth[m]),
            true_buggy=mape(buggy, lambda m: truth[m]),
            reported_buggy=reported,
        )
    return out


# -- report -------------------------------------------------------------------------

def gap_closed(r) -> float:
    return (r.default - r.tournament) / (r.default - r.oracle)


def pct(x: float) -> str:
    return f"{100 * x:5.2f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=50, help="held-out seeds 0..N-1 for robustness")
    ap.add_argument("--max-regret", type=float, default=None,
                    help="fail if tournament MAPE > best fixed engine (hindsight) * (1 + this), any metric/horizon")
    ap.add_argument("--beat-default", action="store_true",
                    help="fail unless the tournament beats the default engine (mean3) on every metric/horizon")
    ap.add_argument("--beat-frozen", action="store_true",
                    help="fail unless the tournament is <= the warm-up pick on every metric/horizon")
    ap.add_argument("--min-holdout-beat-default", type=float, default=None,
                    help="fail if the share of held-out (seed, metric, horizon) cells beating mean3 is below this")
    ap.add_argument("--max-holdout-median-regret", type=float, default=None,
                    help="fail if the median held-out regret vs the hindsight-best fixed engine exceeds this")
    ap.add_argument("--rule", choices=["band", "champion"], default=None,
                    help="override the selection rule (to see the gates go red on a worse rule)")
    ap.add_argument("--window", type=int, default=None, help="override the scoreboard window")
    ap.add_argument("--half-life", type=float, default=None, help="override the recency half-life")
    ap.add_argument("--traps", action="store_true",
                    help="fail unless both trap gates catch their bug and stay silent on the fix")
    a = ap.parse_args()
    failures = []
    if a.rule:
        RUN_KW["rule"] = a.rule
    if a.window:
        RUN_KW["selection_window"] = a.window
    if a.half_life:
        RUN_KW["half_life"] = a.half_life

    print("=" * 78)
    print(f" Forecast Tournament - Eval   (seed {a.seed}, targets {EVAL_FROM}..83 = 48 months)"
          + (f"   rule overrides: {RUN_KW}" if RUN_KW else ""))
    print("=" * 78)
    sc = seed_scores(a.seed)
    for metric in METRICS:
        print(f"\n [{metric}]  MAPE by horizon")
        print(f"   {'horizon':26s} {'tourn.':>7s} {'mean3':>7s} {'frozen pick':>24s} {'best fixed (hindsight)':>32s} "
              f"{'oracle':>7s} {'gap closed':>10s}")
        for _, r in sc[sc.metric == metric].iterrows():
            print(f"   {HORIZON_LABEL[r.h]:26s} {pct(r.tournament):>7s} {pct(r.default):>7s} "
                  f"{r.frozen:>15s} {pct(r.frozen_mape):>7s} {r.best_fixed:>23s} {pct(r.best_fixed_mape):>7s} "
                  f"{pct(r.oracle):>7s} {gap_closed(r):>10.0%}")
    print("   gap closed = share of the mean3 -> oracle error gap the tournament removes")
    print("\n per-engine MAPE (fixed, whole window):")
    print("   " + " ".join(f"{e[:9]:>9s}" for e in ENGINES) + "   metric/h")
    for _, r in sc.iterrows():
        print("   " + " ".join(f"{pct(r[f'mape_{e}']):>9s}" for e in ENGINES) + f"   {r.metric}/h{r.h}")

    for _, r in sc.iterrows():
        tag = f"{r.metric}/h{r.h}"
        if a.max_regret is not None and r.tournament > r.best_fixed_mape * (1 + a.max_regret):
            failures.append(f"{tag}: tournament {pct(r.tournament)} > best fixed {pct(r.best_fixed_mape)} x (1+{a.max_regret})")
        if a.beat_default and not r.tournament < r.default:
            failures.append(f"{tag}: tournament {pct(r.tournament)} does not beat mean3 {pct(r.default)}")
        if a.beat_frozen and not r.tournament <= r.frozen_mape:
            failures.append(f"{tag}: tournament {pct(r.tournament)} worse than frozen pick {pct(r.frozen_mape)}")

    if a.seeds:
        d = holdout(range(a.seeds))
        print("-" * 78)
        print(f" held-out seeds 0..{a.seeds - 1}: mean MAPE and share of seeds where the tournament wins")
        print(f"   {'metric/horizon':16s} {'tourn.':>7s} {'mean3':>7s} {'frozen':>7s} {'best*':>7s} {'oracle':>7s}"
              f"  {'>mean3':>7s} {'>=frozen':>8s} {'>=best*':>8s} {'gap closed':>10s}")
        g = d.groupby(["metric", "h"])
        for (m, h), x in g:
            print(f"   {m + '/h' + str(h):16s} {pct(x.tournament.mean()):>7s} {pct(x.default.mean()):>7s} "
                  f"{pct(x.frozen_mape.mean()):>7s} {pct(x.best_fixed_mape.mean()):>7s} {pct(x.oracle.mean()):>7s}"
                  f"  {x.beat_default.mean():7.0%} {x.beat_frozen.mean():8.0%} {x.beat_best.mean():8.0%}"
                  f" {x.gap_closed.mean():10.0%}")
        share = d.beat_default.mean()
        print(f"   overall: beats mean3 in {share:.1%}, >= frozen pick in {d.beat_frozen.mean():.1%}, "
              f">= best fixed* in {d.beat_best.mean():.1%} of {len(d)} cells; "
              f"median regret vs best* {d.regret.median():+.1%}, mean gap closed {d.gap_closed.mean():.0%}")
        print("   (* best fixed = chosen with hindsight per seed/metric/horizon; not available live)")
        rc = regime_cost(range(a.seeds))
        print(f"   where it loses (h1, MAPE by period around the regime break, seeds 0..{a.seeds - 1}):")
        for (m, per), r in rc.iterrows():
            print(f"     {m:7s} {per:16s} tournament {pct(r.tournament)}   best fixed* {pct(r.best_fixed)}")
        if a.min_holdout_beat_default is not None and share < a.min_holdout_beat_default:
            failures.append(f"held-out beats mean3 in {share:.1%} < {a.min_holdout_beat_default:.0%}")
        if a.max_holdout_median_regret is not None and d.regret.median() > a.max_holdout_median_regret:
            failures.append(f"held-out median regret {d.regret.median():+.1%} > {a.max_holdout_median_regret:+.0%}")

    print("-" * 78)
    print(" trap A - a one-off promotion leaking into rolling windows (h1, seed "
          f"{a.seed}; months right after a promotion)")
    pt = promotion_trap(a.seed)
    for metric, r in pt.items():
        print(f"   [{metric}] {r['n_shadow']} post-promotion target months, {r['n_events']} promotions | "
              f"event-window gate: {r['violations']} promotion reads when leaky "
              f"(strict run {'RAISED' if r['caught'] else 'did NOT raise'}), {r['violations_masked']} when masked")
        for e in (*ROLLING_SHOWN, "tournament"):
            lk, mk = r["rows"][(e, "leaky")], r["rows"][(e, "masked")]
            print(f"     {e:11s} MAPE {pct(lk['mape'])} leaky -> {pct(mk['mape'])} masked   "
                  f"bias {lk['bias']:+.2%} -> {mk['bias']:+.2%}")
        if a.traps and not (r["caught"] and r["violations"] > 0 and r["violations_masked"] == 0):
            failures.append(f"trap A gate failed on {metric}")
        if a.traps and not r["rows"][("mean3", "masked")]["mape"] < r["rows"][("mean3", "leaky")]["mape"]:
            failures.append(f"trap A: masking did not reduce mean3 post-promotion error on {metric}")

    print("-" * 78)
    print(f" trap B - a month cache that freezes the open month's month-to-date figure (seed {a.seed})")
    ct = cache_trap(a.seed)
    for metric, r in ct.items():
        print(f"   [{metric}] {r['frozen_months']} closed months still hold a frozen month-to-date figure; "
              f"final-history gate: {r['violations']} non-final reads with the bug "
              f"(strict run {'RAISED' if r['caught'] else 'did NOT raise'}), "
              f"{r['violations_fixed']} with refresh-on-close; fixed == clean: {r['fixed_identical']}")
        print(f"     tournament MAPE, all horizons:  clean {pct(r['true_clean'])} | buggy, true {pct(r['true_buggy'])} | "
              f"buggy, as the run itself reports {pct(r['reported_buggy'])}")
        if a.traps and not (r["caught"] and r["violations"] > 0 and r["violations_fixed"] == 0
                            and r["fixed_identical"]):
            failures.append(f"trap B gate failed on {metric}")
    print("=" * 78)

    if failures:
        print("FAIL: " + "; ".join(failures))
        raise SystemExit(1)
    print("PASS" if any([a.max_regret is not None, a.beat_default, a.beat_frozen,
                         a.min_holdout_beat_default is not None,
                         a.max_holdout_median_regret is not None, a.traps]) else "(no gates requested)")


if __name__ == "__main__":
    main()
