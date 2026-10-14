"""Tests: the generator plants what it claims, engines cannot see the future, the
selection rule does what it says, both trap gates fire on the bug and stay silent
on the fix, and the headline comparison holds on the shipped seed."""
import copy

import numpy as np
import pytest

from engines import ENGINES, Info, forecast, promo_effect
from scorecard import scorecard
from synth_series import KIND, METRICS, generate
from tournament import (ClosedStore, GateViolation, MonthCache, leaderboard, pick_team, run,
                        score)


@pytest.fixture(scope="module")
def ds():
    return generate(42)


# ---- 1. the generator ---------------------------------------------------------

def test_generator_is_deterministic():
    a, b, c = generate(7), generate(7), generate(8)
    assert a.monthly.equals(b.monthly) and a.daily.equals(b.daily)
    assert not a.monthly.equals(c.monthly)


def test_planted_structure(ds):
    m, cfg = ds.monthly, ds.cfg
    # promotions: margin ratio drops, amount jumps, against the month before
    ev = m.index[m.is_event & (m.index > 0)]
    assert (m.ratio[ev].to_numpy() < m.ratio[ev - 1].to_numpy()).mean() > 0.8
    assert (m.amount[ev].to_numpy() > m.amount[ev - 1].to_numpy()).mean() > 0.8
    # regime break: the ratio level steps up by roughly the planted shift
    clean = m[~m.is_event]
    pre = clean[(clean.idx >= cfg.regime_month - 12) & (clean.idx < cfg.regime_month)].ratio.mean()
    post = clean[(clean.idx >= cfg.regime_month) & (clean.idx < cfg.regime_month + 12)].ratio.mean()
    assert post - pre == pytest.approx(cfg.ratio_regime_shift, abs=0.012)
    # booked = operational + close adjustment
    assert (m.ratio - m.ratio_operational).mean() == pytest.approx(cfg.close_ratio_mu, abs=0.001)


def test_month_to_date(ds):
    i = int(ds.monthly.index[ds.monthly.is_event][0])
    d = int(ds.monthly.days[i])
    # a full month of days is the operational total
    assert ds.mtd("amount", i, d) == pytest.approx(ds.monthly.amount_operational[i])
    assert ds.mtd("ratio", i, d) == pytest.approx(ds.monthly.ratio_operational[i])
    # the baseline view drops promotion days, so in a promotion month it differs
    k = int(ds.monthly.event_end[i])
    assert ds.mtd("ratio", i, k, baseline=True) > ds.mtd("ratio", i, k)


# ---- 2. no look-ahead -----------------------------------------------------------

def _poison_future(ds, T):
    """A copy where everything not knowable on decision day T is replaced."""
    bad = copy.deepcopy(ds)
    k = ds.cfg.asof_day
    for col in ("ratio", "amount"):
        bad.monthly.loc[bad.monthly.idx >= T, col] *= 1.5
    for m, (amt, num, promo) in bad._cum.items():
        if m > T:
            bad._cum[m] = (amt * 3.0, num * 0.5, promo)
        elif m == T:
            amt, num = amt.copy(), num.copy()
            amt[k:] *= 3.0
            num[k:] *= 0.5
            bad._cum[m] = (amt, num, promo)
    return bad


@pytest.mark.parametrize("metric", METRICS)
def test_no_lookahead(ds, metric):
    T = 60
    a, b = run(ds, metric), run(_poison_future(ds, T), metric)
    fa, fb = a.forecasts[a.forecasts.t <= T], b.forecasts[b.forecasts.t <= T]
    assert np.allclose(fa.forecast.to_numpy(), fb.forecast.to_numpy())
    sa, sb = a.selected[a.selected.t <= T], b.selected[b.selected.t <= T]
    assert (sa.team.to_numpy() == sb.team.to_numpy()).all()
    # ...and the poison really does reach later decisions
    assert not np.allclose(a.forecasts[a.forecasts.t > T].forecast, b.forecasts[b.forecasts.t > T].forecast)


def test_engines_read_only_closed_months(ds):
    hist = ds.values("ratio")
    t = 40
    info = Info(t=t, kind=KIND["ratio"], history=hist[:t], mtd=ds.mtd("ratio", t, 15, True),
                mtd_hist=ds.mtd_series("ratio", 15, True)[:t], events=ds.events(), days=ds.days(),
                promo_days=ds.promo_days(), asof_day=15)
    for e in ENGINES:
        for h in range(4):
            f = forecast(e, info, t + h)
            assert np.isfinite(f.value)
            assert all(0 <= m < t for m in f.used + f.overlay_used)
            assert not any(ds.events()[m] for m in f.used), f"{e} read a promotion month"


def test_promotion_overlay_has_the_planted_sign(ds):
    t = 70
    for metric, sign in (("ratio", -1), ("amount", +1)):
        info = Info(t=t, kind=KIND[metric], history=ds.values(metric)[:t], mtd=0.0,
                    mtd_hist=np.zeros(t), events=ds.events(), days=ds.days(),
                    promo_days=ds.promo_days(), asof_day=15)
        effect, used = promo_effect(info)
        assert np.sign(effect) == sign and len(used) > 0


# ---- 3. selection -----------------------------------------------------------------

def test_pick_team_band():
    names = ["a", "b", "c", "d"]
    assert pick_team({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}, names, "band", 0.2, 3) == ["a"]
    assert pick_team({"a": 1.0, "b": 1.1, "c": 1.15, "d": 1.19}, names, "band", 0.2, 3) == ["a", "b", "c"]
    assert pick_team({"a": 1.0, "b": 1.1, "c": 3.0, "d": 4.0}, names, "champion", 0.2, 3) == ["a"]


def test_leaderboard_weights_recent_months():
    errs = {1: {"x": 0.10, "y": 0.0}, 2: {"x": 0.0, "y": 0.10}}
    board = leaderboard(errs, ["x", "y"], window=24, half_life=1.0)
    assert board["x"] < board["y"]                     # y's miss is the more recent one
    flat = leaderboard(errs, ["x", "y"], window=24, half_life=None)
    assert flat["x"] == pytest.approx(flat["y"])


def test_scorecard_matches_the_next_decision(ds):
    month, idx = "2024-06", int(ds.monthly.index[ds.monthly.month == "2024-06"][0])
    text = scorecard(42, "ratio", month)
    res = run(ds, "ratio")
    nxt = res.selected[res.selected.t == idx + 1].set_index("h").team
    runs = [ln.split("runs: ")[1] for ln in text.splitlines() if "next decision runs:" in ln]
    assert runs == [nxt[h].replace("+", " + ") for h in range(4)]


# ---- 4. the trap gates --------------------------------------------------------------

def test_frozen_cache_is_caught_and_the_fix_is_exact(ds):
    with pytest.raises(GateViolation):
        run(ds, "amount", store=MonthCache(ds, "amount", refresh_unclosed=False))
    leaky = run(ds, "amount", store=MonthCache(ds, "amount", refresh_unclosed=False), strict=False)
    assert leaky.gate_final > 0
    fixed = run(ds, "amount", store=MonthCache(ds, "amount", refresh_unclosed=True))
    clean = run(ds, "amount", store=ClosedStore(ds, "amount"))
    assert fixed.gate_final == 0
    assert np.allclose(fixed.selected.forecast, clean.selected.forecast)


def test_promotion_leak_is_caught(ds):
    with pytest.raises(GateViolation):
        run(ds, "ratio", mask_events=False)
    assert run(ds, "ratio", mask_events=False, strict=False).gate_event > 0
    assert run(ds, "ratio").gate_event == 0


# ---- 5. the headline, as a regression guard -------------------------------------------

@pytest.mark.parametrize("metric", METRICS)
def test_tournament_beats_the_default_engine(ds, metric):
    s = score(ds, run(ds, metric))
    assert (s.tournament < s.default).all()
    assert (s.oracle <= s.tournament).all()
