"""Generate a synthetic monthly business series with *known* structure.

Two metrics are produced for every month, bottom-up from daily rows:

  ratio    a margin-style rate (e.g. margin / sales), additive structure
  amount   a monthly total in KRW, multiplicative structure

Ground truth that is planted on purpose, so the eval can check what each
forecasting engine gets right and wrong:

  trend        slow drift in the ratio, steady growth in the amount
  seasonality  a yearly cycle in both
  events       one-off promotion months (planned, so the calendar is known in
               advance): on the promotion days the amount jumps and the ratio
               drops. The effect is confined to that month.
  regime       a structural break at ``regime_month``: the ratio level steps up,
               the amount level steps down, and the seasonal cycle of both
               flattens - permanently. Same-month-last-year engines carry the
               old season for a year; short windows recover in a few months.
  noise        AR(1) month-level noise plus day-level noise
  close        the booked ("closed") monthly value differs from the operational
               daily sum by a month-end close adjustment (accruals, credits)

Daily rows exist for every month, so a forecaster standing on day ``k`` of the
open month can see days 1..k of it (the nowcast input) but not the close.

No real data: every number here is illustrative and generated from a seed.
"""
from __future__ import annotations

import argparse
import calendar
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ("ratio", "amount")
KIND = {"ratio": "additive", "amount": "multiplicative"}
DOW_WEIGHT = np.array([0.86, 0.84, 0.90, 0.95, 1.15, 1.27, 1.03])  # Mon..Sun


@dataclass
class Config:
    n_months: int = 84               # 7 years
    start: str = "2019-01"
    regime_month: int = 50           # structural break (month index)
    asof_day: int = 15               # the decision is taken on this day of the open month
    # ratio (additive)
    ratio_base: float = 0.320
    ratio_trend: float = -0.0002     # per month
    ratio_season_amp: float = 0.010
    ratio_regime_shift: float = 0.025
    regime_season_mult: float = 0.3  # seasonal amplitude after the break (both metrics)
    ratio_ar: float = 0.6
    ratio_ar_sd: float = 0.004
    ratio_daily_sd: float = 0.012
    event_ratio_drop: float = 0.060  # on promotion days
    close_ratio_mu: float = -0.006   # closed - operational
    close_ratio_sd: float = 0.002
    # amount (multiplicative)
    amount_daily_base: float = 40e6  # KRW per day
    amount_growth: float = 0.004     # per month
    amount_season_amp: float = 0.08
    amount_regime_shift: float = -0.10
    amount_ar: float = 0.5
    amount_ar_sd: float = 0.02
    amount_daily_sd: float = 0.08
    event_amount_lift: float = 0.40  # on promotion days
    close_amount_mu: float = -0.012
    close_amount_sd: float = 0.004


@dataclass
class Dataset:
    cfg: Config
    monthly: pd.DataFrame
    daily: pd.DataFrame
    _cum: dict = field(default_factory=dict, repr=False)

    @property
    def n(self) -> int:
        return len(self.monthly)

    def values(self, metric: str) -> np.ndarray:
        """Closed (booked) monthly values: the ground truth a forecast is scored on."""
        return self.monthly[metric].to_numpy(dtype=float)

    def events(self) -> np.ndarray:
        return self.monthly["is_event"].to_numpy(dtype=bool)

    def promo_days(self) -> np.ndarray:
        """Planned promotion days per month (0 outside promotion months)."""
        m = self.monthly
        return np.where(m.is_event, m.event_end - m.event_start + 1, 0).astype(int)

    def days(self) -> np.ndarray:
        return self.monthly["days"].to_numpy(dtype=int)

    def mtd(self, metric: str, month: int, day: int, baseline: bool = False) -> float:
        """Month-to-date operational value from days 1..day of ``month``.

        ratio  -> MTD weighted ratio (numerator / amount)
        amount -> MTD sum (a *partial* total; not comparable to a closed month)

        ``baseline=True`` drops the promotion days seen so far (the calendar is
        known) and, for the amount, rescales the remaining days to a ``day``-day
        equivalent. NaN if every observed day was a promotion day.
        """
        amt, num, promo = (x[:day] for x in self._cum[month])
        if baseline:
            keep = ~promo
            if not keep.any():
                return float("nan")
            scale = len(amt) / keep.sum()
            amt, num = amt[keep], num[keep]
            return float(num.sum() / amt.sum()) if metric == "ratio" else float(amt.sum() * scale)
        return float(num.sum() / amt.sum()) if metric == "ratio" else float(amt.sum())

    def mtd_series(self, metric: str, day: int, baseline: bool = False) -> np.ndarray:
        return np.array([self.mtd(metric, m, day, baseline) for m in range(self.n)])


def _event_calendar(rng: np.random.Generator, n: int) -> list[int]:
    out, m = [], int(rng.integers(4, 8))
    while m < n:
        out.append(m)
        m += int(rng.integers(5, 10))
    return out


def generate(seed: int = 42, cfg: Config | None = None) -> Dataset:
    cfg = cfg or Config()
    rng = np.random.default_rng(seed)
    y0, m0 = (int(x) for x in cfg.start.split("-"))
    events = set(_event_calendar(rng, cfg.n_months))

    ar_r = ar_a = 0.0
    mrows, drows, cum = [], [], {}
    for i in range(cfg.n_months):
        year, month = y0 + (m0 - 1 + i) // 12, (m0 - 1 + i) % 12 + 1
        days = calendar.monthrange(year, month)[1]
        regime = int(i >= cfg.regime_month)

        ar_r = cfg.ratio_ar * ar_r + rng.normal(0, cfg.ratio_ar_sd)
        ar_a = cfg.amount_ar * ar_a + rng.normal(0, cfg.amount_ar_sd)
        phase = 2 * np.pi * (month - 1) / 12
        season = cfg.regime_season_mult if regime else 1.0
        ratio_lvl = (cfg.ratio_base + cfg.ratio_trend * i
                     + season * cfg.ratio_season_amp * np.sin(phase)
                     + cfg.ratio_regime_shift * regime + ar_r)
        amount_lvl = (cfg.amount_daily_base * (1 + cfg.amount_growth) ** i
                      * (1 + season * cfg.amount_season_amp * np.cos(phase - 2 * np.pi * 11 / 12))
                      * (1 + cfg.amount_regime_shift * regime) * np.exp(ar_a))

        ev_start = ev_end = 0
        if i in events:
            length = int(rng.integers(9, 17))
            ev_start = int(rng.integers(1, days - length + 2))
            ev_end = ev_start + length - 1

        amts, nums = np.empty(days), np.empty(days)
        for d in range(1, days + 1):
            wd = calendar.weekday(year, month, d)
            on_event = ev_start <= d <= ev_end and ev_start > 0
            a = amount_lvl * DOW_WEIGHT[wd] * np.exp(rng.normal(0, cfg.amount_daily_sd))
            r = ratio_lvl + rng.normal(0, cfg.ratio_daily_sd)
            if on_event:
                a *= 1 + cfg.event_amount_lift
                r -= cfg.event_ratio_drop
            amts[d - 1], nums[d - 1] = a, a * r
            drows.append(dict(month_idx=i, date=f"{year:04d}-{month:02d}-{d:02d}", day=d,
                              amount=round(a), numerator=round(a * r), promo=on_event))
        promo = np.zeros(days, dtype=bool)
        if ev_start:
            promo[ev_start - 1:ev_end] = True
        cum[i] = (amts, nums, promo)

        amount_op = amts.sum()
        ratio_op = nums.sum() / amount_op
        ratio = ratio_op + rng.normal(cfg.close_ratio_mu, cfg.close_ratio_sd)
        amount = amount_op * (1 + rng.normal(cfg.close_amount_mu, cfg.close_amount_sd))
        mrows.append(dict(idx=i, month=f"{year:04d}-{month:02d}", days=days,
                          is_event=i in events, event_start=ev_start, event_end=ev_end,
                          regime=regime, ratio=ratio, amount=amount,
                          ratio_operational=ratio_op, amount_operational=amount_op))

    return Dataset(cfg, pd.DataFrame(mrows), pd.DataFrame(drows), cum)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=Path("data"))
    a = ap.parse_args()

    ds = generate(seed=a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    ds.monthly.to_csv(a.out / "monthly.csv", index=False, encoding="utf-8-sig")
    ds.daily.to_csv(a.out / "daily.csv", index=False, encoding="utf-8-sig")
    m = ds.monthly
    print(f"[synth] {len(m)} months {m.month.iloc[0]}..{m.month.iloc[-1]}, "
          f"{int(m.is_event.sum())} promotion months, regime break at {m.month.iloc[ds.cfg.regime_month]}, "
          f"{len(ds.daily):,} daily rows -> {a.out}/")


if __name__ == "__main__":
    main()
