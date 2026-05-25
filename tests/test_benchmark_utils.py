import math

from qml._benchmark_utils import mean_std, summary_stats, timing_from_result


def test_mean_std_handles_empty_singleton_and_multiple_values() -> None:
    empty = mean_std([])
    assert math.isnan(empty["mean"])
    assert math.isnan(empty["std"])

    assert mean_std([3.0]) == {"mean": 3.0, "std": 0.0}
    assert mean_std([1.0, 3.0]) == {"mean": 2.0, "std": 1.0}


def test_summary_stats_adds_sample_count_and_confidence_interval() -> None:
    stats = summary_stats([1.0, 3.0])

    assert stats["n"] == 2
    assert stats["mean"] == 2.0
    assert stats["std"] == 1.0
    assert stats["ci95_low"] < stats["mean"] < stats["ci95_high"]


def test_timing_from_result_preserves_model_timing_and_fills_total() -> None:
    timing = timing_from_result({"timing": {"fit_seconds": 0.25}}, runtime_seconds=1.5)

    assert timing == {"fit_seconds": 0.25, "total_seconds": 1.5}
    assert timing_from_result({}, runtime_seconds=2.0) == {"total_seconds": 2.0}
