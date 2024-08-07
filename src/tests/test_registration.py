from typing import cast

import gymnasium
import pytest

from src.dynamics.env import DynamicEnv
from src.dynamics.registration import CreditEnvCreator, make_env
from src.dynamics.simulator import State
from src.loader.credit import CreditData


@pytest.mark.parametrize("memory", [True, False])
def test_env_registration(memory: bool) -> None:
    assert CreditEnvCreator.ID in gymnasium.registry
    ds = CreditData(seed=0)
    initial_state = State(features=ds.features, labels=ds.labels)
    gymnasium.make(CreditEnvCreator.ID, initial_state=initial_state)
    env = CreditEnvCreator.as_env(initial_state=initial_state, memory=memory)
    env = cast(DynamicEnv, env)
    assert env.simulator.memory is memory


@pytest.mark.parametrize(
    "epsilon, memory",
    [
        ({a: 0.1 for a in range(2)}, True),
        ({a: 0.0 for a in range(2)}, False),
        ({a: 1.0 for a in range(2)}, True),
    ],
)
def test_make_env(
    mock_state: State,  # noqa: F811
    epsilon: dict[int, float],
    memory: bool,
):
    # Arrange - mock_state fixture provides initial_state
    # Act
    env = make_env(initial_state=mock_state, epsilon=epsilon, memory=memory)
    assert env is not None
