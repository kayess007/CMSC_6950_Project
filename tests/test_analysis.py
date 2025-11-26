import pytest
import numpy as np
import pandas as pd

from analysis import (
    compute_daily_statistics,
    identify_extremes,
    monthly_summary,
)


@pytest.mark.parametrize(
    "temps, expected_mean, expected_median, expected_std",
    [
        ([0.0, 1.0, 2.0, 3.0], 1.5, 1.5, np.std([0, 1, 2, 3], ddof=1)),
        ([5.0, np.nan, 15.0], 10.0, 10.0, np.std([5.0, 15.0], ddof=1)),
    ],
)
def test_compute_daily_statistics(temps, expected_mean, expected_median, expected_std):
    df = pd.DataFrame({"temp": temps})
    mean_val, median_val, std_val = compute_daily_statistics(df, "temp")
    assert mean_val == pytest.approx(expected_mean)
    assert median_val == pytest.approx(expected_median)
    assert std_val == pytest.approx(expected_std)


@pytest.mark.parametrize("temps", [
    ([np.nan, np.nan]),
    ([np.nan]),
    ([]),
])
def test_compute_daily_statistics_nan_cases(temps):
    df = pd.DataFrame({"temp": temps})
    mean_val, median_val, std_val = compute_daily_statistics(df, "temp")
    assert np.isnan(mean_val)
    assert np.isnan(median_val)
    assert np.isnan(std_val)


@pytest.mark.parametrize(
    "temps, threshold, expected_count",
    [
        ([10.0, 11.0, 9.5, 40.0], 2.0, 1),  # 40 is the only extreme
        ([5.0, 5.0, 5.0], 2.0, 0),          # zero std → empty
        ([1.0, 2.0, 3.0], 5.0, 0),          # no values far enough away
    ],
)
def test_identify_extremes(temps, threshold, expected_count):
    df = pd.DataFrame({"temp": temps})
    extremes = identify_extremes(df, "temp", threshold_std=threshold)
    assert len(extremes) == expected_count


def test_identify_extremes_correct_value():
    df = pd.DataFrame({"temp": [10.0, 11.0, 9.5, 40.0]})
    extremes = identify_extremes(df, "temp", threshold_std=2.0)
    assert len(extremes) == 1
    assert extremes["temp"].iloc[0] == 40.0


@pytest.mark.parametrize(
    "dates, temps, expected_months",
    [
        (
            pd.to_datetime(["2024-01-01", "2024-01-15", "2024-01-31",
                            "2024-02-01", "2024-02-28"]),
            [0.0, 2.0, 4.0, 10.0, 12.0],
            2,
        ),
        (
            pd.to_datetime(["2024-01-10", "2024-01-20"]),
            [5.0, 7.0],
            1,
        ),
    ],
)
def test_monthly_summary_grouping(dates, temps, expected_months):
    df = pd.DataFrame({"date": dates, "temp": temps})
    result = monthly_summary(df, "date", "temp")
    assert len(result) == expected_months


def test_monthly_summary_values():
    dates = pd.to_datetime(
        ["2024-01-01", "2024-01-15", "2024-01-31",
         "2024-02-01", "2024-02-28"]
    )
    temps = [0.0, 2.0, 4.0, 10.0, 12.0]
    df = pd.DataFrame({"date": dates, "temp": temps})
    result = monthly_summary(df, "date", "temp")

    jan = result.loc[result["month"] == pd.Timestamp("2024-01-31")]
    feb = result.loc[result["month"] == pd.Timestamp("2024-02-29")]

    assert not jan.empty
    assert not feb.empty

    assert jan["mean"].iloc[0] == pytest.approx((0 + 2 + 4) / 3)
    assert jan["min"].iloc[0] == 0.0
    assert jan["max"].iloc[0] == 4.0
    assert jan["count"].iloc[0] == 3

    assert feb["mean"].iloc[0] == pytest.approx((10 + 12) / 2)
    assert feb["min"].iloc[0] == 10.0
    assert feb["max"].iloc[0] == 12.0
    assert feb["count"].iloc[0] == 2
