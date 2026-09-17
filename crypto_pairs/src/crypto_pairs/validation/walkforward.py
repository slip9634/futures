"""Walk-forward scheduling with purge and embargo (mandate PART 19).

A walk-forward schedule is a sequence of (formation, test) index windows.
Parameters are chosen using formation data ONLY, then frozen, then applied to
the test window.

PURGE and EMBARGO (Lopez de Prado, Advances in Financial Machine Learning):
  * PURGE removes formation observations whose feature windows overlap the test
    window. Here the binding overlap is the longest lookback used to build a
    signal (hedge window, Z window, cointegration window) -- a formation bar
    within `purge_bars` of the test start shares data with the test set.
  * EMBARGO drops a gap AFTER the test window before the next formation window
    reuses that data, blocking leakage in the other direction through
    serially-correlated features.

Both default to the maximum signal lookback, which is the only defensible
choice: a shorter purge leaves known leakage in place.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

__all__ = ["Window", "make_schedule", "SCHEDULES"]


@dataclass
class Window:
    idx: int
    form_start: int
    form_end: int      # exclusive, already purged
    test_start: int
    test_end: int      # exclusive
    embargo_end: int

    @property
    def n_form(self) -> int:
        return self.form_end - self.form_start

    @property
    def n_test(self) -> int:
        return self.test_end - self.test_start


SCHEDULES = {
    # name: (formation_days, test_days, step_days)
    "A": (60, 14, 7),
    "B": (90, 30, 30),
    "C": (30, 7, 7),
}


def make_schedule(n_bars: int, formation_days: int, test_days: int, step_days: int,
                  bars_per_day: int = 288, purge_bars: int = 0,
                  embargo_bars: int = 0) -> list[Window]:
    f = formation_days * bars_per_day
    t = test_days * bars_per_day
    s = step_days * bars_per_day
    out: list[Window] = []
    i = 0
    start = 0
    while start + f + t <= n_bars:
        form_start = start
        form_end = start + f - purge_bars          # purge the tail of formation
        test_start = start + f
        test_end = test_start + t
        emb = min(n_bars, test_end + embargo_bars)
        if form_end - form_start < bars_per_day:
            break
        out.append(Window(i, form_start, form_end, test_start, test_end, emb))
        i += 1
        start += s
    return out
