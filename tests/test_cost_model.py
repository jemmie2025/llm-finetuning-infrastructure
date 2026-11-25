import pytest

from scripts.cost_model import estimate_cost
from scripts.savings_model import calculate_savings


def test_cost_calculation():
    hours, cost = estimate_cost(360000, 1, 100, 2, 0)
    assert hours == pytest.approx(1.0)
    assert cost == pytest.approx(2.0)


def test_runtime_overhead():
    hours, cost = estimate_cost(360000, 1, 100, 2, 0.15)
    assert hours == pytest.approx(1.15)
    assert cost == pytest.approx(2.30)


def test_multiple_epochs():
    hours, cost = estimate_cost(360000, 3, 100, 2, 0)
    assert hours == pytest.approx(3.0)
    assert cost == pytest.approx(6.0)


@pytest.mark.parametrize("tokens,epochs,throughput,rate,overhead", [
    (0, 1, 100, 2, 0),
    (100, 0, 100, 2, 0),
    (100, 1, 0, 2, 0),
    (100, 1, 100, -1, 0),
    (100, 1, 100, 2, -0.1),
    (float("nan"), 1, 100, 2, 0),
    (100, 1, float("inf"), 2, 0),
])
def test_invalid_cost_inputs(tokens, epochs, throughput, rate, overhead):
    with pytest.raises(ValueError):
        estimate_cost(tokens, epochs, throughput, rate, overhead)


@pytest.mark.parametrize("fraction,expected", [
    (0.20, 17.80),
    (0.40, 35.60),
    (0.35, 31.15),
    (0, 0),
    (1, 89),
])
def test_monthly_savings(fraction, expected):
    assert calculate_savings(89, fraction) == expected


@pytest.mark.parametrize("baseline,fraction", [
    (-89, 0.20),
    (89, -0.10),
    (89, 1.10),
])
def test_invalid_savings_inputs(baseline, fraction):
    with pytest.raises(ValueError):
        calculate_savings(baseline, fraction)
