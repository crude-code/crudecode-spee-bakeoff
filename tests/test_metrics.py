import numpy as np

from forecast_benchmark.metrics import bias, log_error, mape, spee_score


def test_mape_perfect_forecast_is_zero():
    a = np.array([100.0, 200.0, 300.0])
    assert mape(a, a) == 0.0


def test_mape_returns_none_for_all_zero_actuals():
    a = np.zeros(5)
    f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert mape(a, f) is None


def test_bias_direction_over_forecast_positive():
    a = np.array([100.0, 100.0])
    f = np.array([120.0, 120.0])
    b = bias(a, f)
    assert b is not None and b > 0


def test_bias_direction_under_forecast_negative():
    a = np.array([100.0, 100.0])
    f = np.array([80.0, 80.0])
    b = bias(a, f)
    assert b is not None and b < 0


def test_log_error_drops_nonpositive_pairs():
    a = np.array([100.0, -5.0, 50.0])
    f = np.array([110.0, 10.0, 0.0])
    le = log_error(a, f)
    # only the (100, 110) pair is valid: -5 actual and 0 forecast both drop
    assert le is not None
    assert len(le) == 1


def test_log_error_none_when_nothing_valid():
    a = np.array([-1.0, -2.0])
    f = np.array([1.0, 1.0])
    assert log_error(a, f) is None


def test_spee_score_zero_for_perfect_forecast():
    a = np.array([100.0, 150.0, 200.0, 250.0])
    score = spee_score(a, a)
    assert score is not None
    assert abs(score) < 1e-9


def test_spee_score_none_with_insufficient_points():
    a = np.array([100.0])
    f = np.array([100.0])
    assert spee_score(a, f) is None


def test_metrics_skip_nan_actual_months():
    a = np.array([100.0, np.nan, 200.0])
    f = np.array([100.0, 999.0, 200.0])
    # the NaN actual month should not corrupt a perfect-elsewhere forecast
    assert mape(a, f) == 0.0


def test_spee_decomposition_reports_bias_and_spread_components():
    from forecast_benchmark.metrics import spee_decomposition

    a = np.array([100.0, 100.0, 100.0, 100.0])
    f = np.array([90.0, 100.0, 110.0, 120.0])
    d = spee_decomposition(a, f)
    assert d is not None
    assert d.score == d.bias_component + d.spread_component
    assert abs((d.bias_share + d.spread_share) - 1.0) < 1e-12
    assert d.n_log_errors == 4


def test_metric_epsilon_penalizes_zero_forecast_without_changing_forecast():
    from forecast_benchmark.metrics import spee_decomposition

    actual = np.array([100.0, 100.0, 100.0])
    forecast = np.array([100.0, 0.0, 100.0])
    original = forecast.copy()
    d = spee_decomposition(actual, forecast, forecast_zero_policy="epsilon")
    assert d is not None
    assert d.n_nonpositive_forecast_replaced == 1
    assert d.score > 0
    assert np.array_equal(forecast, original)


def test_metric_zero_policy_can_fail_closed():
    import pytest
    from forecast_benchmark.metrics import log_error

    with pytest.raises(ValueError, match="nonpositive forecasts"):
        log_error(np.array([1.0, 1.0]), np.array([1.0, 0.0]), forecast_zero_policy="error")


def test_cumulative_log_error_is_one_error_per_well_phase():
    from forecast_benchmark.metrics import cumulative_log_error

    actual = np.array([50.0, 50.0])
    forecast = np.array([40.0, 60.0])
    assert cumulative_log_error(actual, forecast) == 0.0


def test_cumulative_zero_forecast_uses_metric_only_epsilon():
    from forecast_benchmark.metrics import cumulative_log_error_with_audit

    actual = np.array([10.0, 20.0])
    forecast = np.zeros(2)
    value, audit = cumulative_log_error_with_audit(actual, forecast, forecast_zero_policy="epsilon")
    assert value is not None and value < 0
    assert audit["n_nonpositive_forecast_replaced"] == 1
