"""Figure helpers: a gap in the data must stay a gap, and labels must not sit on each other."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from chess_elo_drift.analysis import plots


def drawn(years, values, low=None, high=None):
    figure, axis = plt.subplots()
    low = values if low is None else low
    high = values if high is None else high
    end = plots._line_with_band(axis, years, values, low, high, "#2a78d6", "1500")
    return figure, axis, end


def test_a_line_breaks_across_years_with_no_value():
    figure, axis, end = drawn([2019, 2020, 2021, 2026], [1.0, 2.0, 3.0, 4.0])
    line = axis.lines[0]
    x, y = line.get_xdata(), line.get_ydata()
    assert list(x) == list(range(2019, 2027))
    assert np.isnan(y[3:7]).all() and np.isfinite(y[:3]).all() and y[7] == 4.0
    assert end == (2026, 4.0, "1500")
    plt.close(figure)


def test_a_band_is_not_carried_across_a_gap():
    figure, axis, _ = drawn([2019, 2020, 2024, 2025], [1.0, 2.0, 3.0, 4.0], low=[0.0, 1.0, 2.0, 3.0], high=[2.0, 3.0, 4.0, 5.0])
    polygons = axis.collections[0].get_paths()
    assert len(polygons) == 2
    spans = sorted((path.vertices[:, 0].min(), path.vertices[:, 0].max()) for path in polygons)
    assert spans[0][1] <= 2020 and spans[1][0] >= 2024
    plt.close(figure)


def test_a_lone_year_keeps_its_interval_as_a_bar():
    figure, axis, _ = drawn([2014, 2026], [1.0, 2.0], low=[0.5, 1.0], high=[1.5, 3.0])
    assert len(axis.collections) == 2  # the band, and the bars for the two isolated years
    plt.close(figure)


def test_end_labels_that_would_overlap_are_pushed_apart_and_others_left_alone():
    placed = plots._spread_labels([100.0, 103.0, 300.0], plots.MIN_LABEL_GAP_POINTS)
    assert placed[1] - placed[0] >= plots.MIN_LABEL_GAP_POINTS - 1e-9
    assert placed[2] == 300.0
    assert np.mean(placed[:2]) == np.mean([100.0, 103.0])
    assert plots._spread_labels([], 10.0) == []


def test_the_nudge_keeps_the_order_of_the_labels():
    placed = plots._spread_labels([5.0, 1.0, 3.0], 4.0)
    assert np.argsort(placed).tolist() == [1, 2, 0]
