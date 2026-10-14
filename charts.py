"""Render the two README charts.

docs/tournament.png   one seed, the ratio metric: the closed series with its
                      promotion months and the regime break, the h1 forecast the
                      tournament actually used, and - underneath - which engine led
                      each horizon's race month by month
docs/scoreboard.png   held-out seeds 0..49: mean MAPE by horizon for the tournament,
                      the default engine, the frozen warm-up pick, the best fixed
                      engine in hindsight, and the per-month oracle

Static PNGs so they render on GitHub.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from engines import ENGINES  # noqa: E402
from eval import holdout  # noqa: E402
from synth_series import generate  # noqa: E402
from tournament import EVAL_FROM, HORIZONS, run  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#ebeae6"
# fixed categorical order; colour follows the engine, never its rank
ENGINE_COLOR = {"nowcast": "#2a78d6", "mean3": "#eb6834", "mean12": "#1baf7a",
                "seasonal": "#eda100", "seasonal_growth": "#e87ba4", "ses": "#008300",
                "trend": "#4a3aa7", "last": "#e34948"}
SERIES = [("tournament", "tournament", "#2a78d6", "-"),
          ("default", "mean3 (default)", "#eb6834", "-"),
          ("frozen_mape", "frozen warm-up pick", "#1baf7a", "-"),
          ("best_fixed_mape", "best fixed (hindsight)", "#eda100", "-"),
          ("oracle", "oracle (per month)", MUTED, "--")]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def tournament_chart(seed: int, out: Path) -> None:
    ds = generate(seed)
    res = run(ds, "ratio")
    m = ds.monthly
    x = np.arange(ds.n)
    fig = plt.figure(figsize=(11, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.35], hspace=0.12)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    for i in np.flatnonzero(ds.events()):
        ax.axvspan(i - 0.5, i + 0.5, color="#f0efec", lw=0)
    reg = ds.cfg.regime_month
    for a in (ax, ax2):
        a.axvline(reg - 0.5, color=MUTED, lw=1, ls=(0, (3, 3)))
    ax.axvline(EVAL_FROM - 0.5, color=GRID, lw=1)

    ax.plot(x, 100 * m.ratio, color=INK, lw=2, label="closed actual")
    s = res.selected[res.selected.h == 1]
    ax.plot(s.target, 100 * s.forecast, color=ENGINE_COLOR["nowcast"], lw=2,
            label="tournament forecast, h1 (made a month ahead)")
    ax.set_ylabel("ratio (%)", fontsize=9, color=MUTED)
    ax.set_xlim(-0.5, ds.n - 0.5)
    ax.set_xticks([])
    _style(ax)
    ytop = ax.get_ylim()[1]
    ax.text(reg, ytop, " regime break", fontsize=8, color=MUTED, va="top")
    ax.text(EVAL_FROM, ytop, " scored window", fontsize=8, color=MUTED, va="top")
    ax.legend(handles=[*ax.get_legend_handles_labels()[0], Patch(color="#f0efec", label="promotion month")],
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8, frameon=False, ncol=3)
    ax.set_title("Which engine forecasts next month? A walk-forward tournament decides (synthetic)",
                 fontsize=13, fontweight="bold", color=INK, loc="left", pad=24)

    # leader of each horizon's race, per target month
    for row, h in enumerate(HORIZONS):
        sh = res.selected[res.selected.h == h]
        for _, r in sh.iterrows():
            team = r.team.split("+")
            ax2.add_patch(plt.Rectangle((r.target - 0.5, row - 0.4), 1, 0.8, lw=0,
                                        color=ENGINE_COLOR[team[0]]))
            if len(team) > 1:
                ax2.plot(r.target, row, marker="o", ms=2.2, color=SURFACE, lw=0)
    ax2.set_xlim(-0.5, ds.n - 0.5)
    ax2.set_ylim(len(HORIZONS) - 0.5, -0.5)
    ax2.set_yticks(range(len(HORIZONS)))
    ax2.set_yticklabels([f"h{h}" for h in HORIZONS])
    ticks = [i for i in range(ds.n) if m.month[i].endswith("-01")]
    ax2.set_xticks(ticks)
    ax2.set_xticklabels([m.month[i][:4] for i in ticks])
    _style(ax2)
    ax2.grid(False)
    ax2.set_ylabel("race leader", fontsize=9, color=MUTED)
    used = [e for e in ENGINES if res.selected.team.str.split("+").str[0].eq(e).any()]
    ax2.legend(handles=[Patch(color=ENGINE_COLOR[e], label=e) for e in used], loc="upper center",
               bbox_to_anchor=(0.5, -0.22), ncol=len(used), fontsize=8, frameon=False)
    ax2.text(0, -0.62, "white dot = the leader ran with tied engines averaged in;"
             " the default engine (mean3) runs until three targets per horizon have closed",
             transform=ax2.transAxes, fontsize=7.5, color=MUTED)
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def scoreboard_chart(seeds: int, out: Path) -> None:
    d = holdout(range(seeds))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), facecolor=SURFACE)
    for ax, metric in zip(axes, ("ratio", "amount")):
        g = d[d.metric == metric].groupby("h").mean(numeric_only=True)
        for col, label, color, ls in SERIES:
            ax.plot(g.index, 100 * g[col], color=color, lw=2, ls=ls, marker="o", ms=5, label=label)
        ax.set_xticks(list(HORIZONS))
        ax.set_xticklabels(["h0\nopen month", "h1\nnext", "h2", "h3"])
        ax.set_xlim(-0.2, 3.2)
        ax.set_ylim(0, None)
        ax.set_ylabel("mean MAPE (%)", fontsize=9, color=MUTED)
        ax.set_title(f"{metric}", fontsize=11, color=INK, loc="left", fontweight="bold")
        _style(ax)
    fig.suptitle(f"Forecast error by horizon, {seeds} held-out seeds (synthetic)", x=0.06, ha="left",
                 fontsize=13, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(0.055, 0.945),
               ncol=len(SERIES), fontsize=8.5, frameon=False)
    fig.savefig(out, dpi=130, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seeds", type=int, default=50)
    a = ap.parse_args()
    docs = Path(__file__).with_name("docs")
    docs.mkdir(exist_ok=True)
    tournament_chart(a.seed, docs / "tournament.png")
    scoreboard_chart(a.seeds, docs / "scoreboard.png")
    print(f"wrote {docs / 'tournament.png'} and {docs / 'scoreboard.png'}")


if __name__ == "__main__":
    main()
