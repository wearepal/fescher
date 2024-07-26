import numpy as np
import pytest

from src.dynamics.response import LinearResponse
from src.dynamics.simulator import Rollout, Simulator
from src.dynamics.state import State


def test_simulator(mock_state: State) -> None:
    action = np.random.default_rng(0).uniform(low=0, high=1, size=(mock_state.num_features,))
    run = Simulator(
        response=LinearResponse(epsilon=1.0),
        memory=False,
    ).simulate(
        state=mock_state,
        action=action,
        start_time=0,
        steps=30,
    )
    assert len(run.states) == len(run.times)
    assert run.initial_state is mock_state
    assert np.allclose(1, run.times[1] - run.times[0])


@pytest.fixture
def mock_states():
    state1 = State(
        features=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
        labels=np.array([1, 0], dtype=np.int64),
    )
    state2 = State(
        features=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64),
        labels=np.array([1, 1], dtype=np.int64),
    )
    state3 = State(
        features=np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float64),
        labels=np.array([0, 1], dtype=np.int64),
    )
    return [state1, state2, state3]


@pytest.fixture
def mock_times():
    return [0, 5, 10]


@pytest.fixture
def valid_rollout(mock_states: list[State], mock_times: list[int]) -> Rollout:
    return Rollout(states=mock_states, times=mock_times)


def test_rollout_init(valid_rollout: Rollout):
    assert len(valid_rollout.states) == 3, "Rollout should have 3 states"
    assert len(valid_rollout.times) == 3, "Rollout should have 3 times"


@pytest.mark.parametrize(
    "states, times, expected_exception",
    [
        (
            [
                State(
                    features=np.array([[1.0, 2.0]], dtype=np.float64),
                    labels=np.array([1], dtype=np.int64),
                )
            ],
            [],
            ValueError,
        ),
        (
            [
                State(
                    features=np.array([[1.0, 2.0]], dtype=np.float64),
                    labels=np.array([1], dtype=np.int64),
                )
            ],
            [0, 5],
            ValueError,
        ),
    ],
)
def test_rollout_init_exceptions(
    states: list[State], times: list[int], expected_exception: type[RuntimeError]
):
    with pytest.raises(expected_exception):
        Rollout(states=states, times=times)


@pytest.mark.parametrize(
    "time, expected_state_index", [(0, 0), (2, 0), (5, 1), (8, 2), (10, 2), (15, 2)]
)
def test_getitem(valid_rollout: Rollout, time: int, expected_state_index: int):
    state = valid_rollout[time]
    assert (
        state == valid_rollout.states[expected_state_index]
    ), f"State at time {time} should be state at index {expected_state_index}"


def test_initial_state(valid_rollout: Rollout):
    assert (
        valid_rollout.initial_state == valid_rollout.states[0]
    ), "Initial state should be the first state in the list"


def test_initial_time(valid_rollout: Rollout):
    assert (
        valid_rollout.initial_time == valid_rollout.times[0]
    ), "Initial time should be the first time in the list"


def test_final_state(valid_rollout: Rollout):
    assert (
        valid_rollout.final_state == valid_rollout.states[-1]
    ), "Final state should be the last state in the list"


def test_final_time(valid_rollout: Rollout):
    assert (
        valid_rollout.final_time == valid_rollout.times[-1]
    ), "Final time should be the last time in the list"
