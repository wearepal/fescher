import numpy as np
import pytest

from src.dynamics.env import DynamicEnv
from src.models.lr import Model
from src.repeated_risk_min import repeated_risk_minimization


@pytest.mark.parametrize(
    "num_steps, l2_penalty, expected_loss_start, expected_loss_end, expected_acc_start, expected_acc_end, expected_theta_gap",
    [
        (1, 0.1, [0.93], [0.6], [0.5], [0.5], [0.28284271247461906]),
        (2, 0.2, [0.93, 1.44], [0.6, 0.6], [0.5, 0.5], [0.5, 0.5], [0.28284271247461906, 0.0]),
    ],
    ids=["a", "b"],
)
def test_repeated_risk_minimization(
    mock_env: DynamicEnv,
    mock_model: Model,
    num_steps: int,
    l2_penalty: float,
    expected_loss_start: list[float],
    expected_loss_end: list[float],
    expected_acc_start: list[float],
    expected_acc_end: list[float],
    expected_theta_gap: list[float],
):
    # Act
    result = repeated_risk_minimization(
        env=mock_env, num_steps=num_steps, lr=mock_model, l2_penalty=l2_penalty
    )

    # Assert
    np.testing.assert_almost_equal(result.loss_start, expected_loss_start, decimal=2)
    assert result.loss_end == expected_loss_end
    assert result.acc_start == expected_acc_start
    assert result.acc_end == expected_acc_end
    np.testing.assert_almost_equal(result.theta_gap, expected_theta_gap, decimal=2)
