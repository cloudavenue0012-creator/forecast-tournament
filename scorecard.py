"""The month-close scorecard - what lands in the inbox when a month's books close.

For the month that just closed it lists, per horizon, what every engine had
forecast for it and how far off each was, then the updated scoreboard and the
team that will forecast the next decision. In production this text is the body
of a scheduled e-mail; here it prints to stdout.

    python scorecard.py --month 2024-06
    python scorecard.py --month 2024-06 --metric amount
"""
from __future__ import annotations

import argparse

import numpy as np

from engines import ENGINES
from synth_series import METRICS, generate
from tournament import HORIZON_LABEL, HORIZONS, RULE, leaderboard, pick_team, run


def scorecard(seed: int, metric: str, month: str) -> str:
    ds = generate(seed)
    idx = int(ds.monthly.index[ds.monthly.month == month][0])
    res = run(ds, metric)
    truth = ds.values(metric)
    fc = res.forecasts.copy()
    fc["ape"] = (fc.forecast - truth[fc.target]).abs() / np.abs(truth[fc.target])
    fmt = (lambda v: f"{100 * v:6.2f}%") if metric == "ratio" else (lambda v: f"{v / 1e6:9,.0f}M")
    names = list(ENGINES)

    lines = [f"[{metric}] month {month} closed at {fmt(truth[idx]).strip()}"
             f"{'  (promotion month)' if ds.events()[idx] else ''}", ""]
    for h in HORIZONS:
        f = fc[(fc.h == h) & (fc.target == idx)].set_index("engine")
        if f.empty:
            continue
        sel = res.selected[(res.selected.h == h) & (res.selected.target == idx)].iloc[0]
        sel_ape = abs(sel.forecast - truth[idx]) / abs(truth[idx])
        lines.append(f"  {HORIZON_LABEL[h]} - forecast at {ds.monthly.month[idx - h]} day {ds.cfg.asof_day}")
        lines.append(f"    used: {sel.team:30s} {fmt(sel.forecast)}   error {100 * sel_ape:5.2f}%")
        for e in sorted(names, key=lambda e: f.ape[e]):
            lines.append(f"      {e:16s} {fmt(f.forecast[e])}   error {100 * f.ape[e]:5.2f}%")
        # the scoreboard as it stands now that this month is closed
        errs = {}
        for _, r in fc[(fc.h == h) & (fc.target <= idx)].iterrows():
            errs.setdefault(int(r.target), {})[r.engine] = r.ape
        board = leaderboard(errs, names, RULE["selection_window"], RULE["half_life"])
        team = pick_team(board, names, "band", RULE["band"], RULE["top_k"])
        top = ", ".join(f"{e} {100 * board[e]:.2f}%" for e in sorted(names, key=board.get)[:3])
        lines.append(f"    scoreboard (weighted MAPE, last {RULE['selection_window']}): {top}")
        lines.append(f"    next decision runs: {' + '.join(team)}")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--month", default="2024-06")
    ap.add_argument("--metric", choices=METRICS, default="ratio")
    a = ap.parse_args()
    print(scorecard(a.seed, a.metric, a.month))


if __name__ == "__main__":
    main()
