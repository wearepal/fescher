import copy
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

import numpy as np

from src.conftest import TESTING
from src.dynamics.response import ResponseFn
from src.dynamics.state import State
from src.types import Action

__all__ = ["simulate", "Rollout", "Simulator"]


@dataclass(kw_only=True)
class Rollout:
    r"""Encapsulate a trajectory from a dynamical system simulator.

    :param states: Sequence of states :math:`x_{t_1}, x_{t_2}, \dots` produced by the simulator.
    :param times: Sequence of sampled times :math:`{t_1},{t_2}, \dots` at which the states were recorded.
    """

    states: list[State]
    times: list[int]

    def __post_init__(self) -> None:
        """Error checking called after the constructor."""
        if len(self.states) != len(self.times):
            msg = (
                "Input states and times must be the same length!"
                f" {len(self.states)} \neq {len(self.times)}",
            )
            raise ValueError(msg)

    def __getitem__(self, time: int) -> State:
        """Return the state closest to the given time."""
        time_index = np.argmin(np.abs(np.array(self.times) - time))
        return self.states[time_index]

    @property
    def initial_state(self) -> State:
        """Return initial state of the run."""
        return self.states[0]

    @property
    def initial_time(self) -> int:
        """Return the initial time of the run."""
        return self.times[0]

    @property
    def final_state(self) -> State:
        """Return the final state of the run."""
        return self.states[-1]

    @property
    def final_time(self) -> int:
        """Return the final time of the run."""
        return self.times[-1]


def simulate(
    *,
    state: State,
    steps: int,
    response: ResponseFn,
    action: Action,
    start_time: int = 0,
    memory: bool = False,
) -> Rollout:
    """Simulate a run of the Credit model."""
    # Iterate the discrete dynamics
    times = [start_time]
    initial_state = state
    states = [initial_state]
    state = copy.deepcopy(state)
    for time in range(start_time, start_time + steps):
        state = state if memory else initial_state
        features = response(features=state.features, action=action)
        state = replace(state, features=features)
        states.append(state)
        times.append(time + 1)

    return Rollout(states=states, times=times)


R = TypeVar("R", bound=ResponseFn)


@dataclass(kw_only=True)
class Simulator(Generic[R]):
    response: R
    memory: bool = False

    def __call__(self, *, state: State, action: Action, steps: int, start_time: int = 0) -> Rollout:
        return simulate(
            state=state,
            steps=steps,
            action=action,
            response=self.response,
            memory=self.memory,
            start_time=start_time,
        )


if TESTING:
    from src.dynamics.response import LinearResponse

    def test_simulator() -> None:
        from src.loader.credit import CreditData

        ds = CreditData(seed=0)
        initial_state = State(features=ds.features, labels=ds.labels)
        action = np.random.uniform(low=0, high=1, size=(initial_state.num_features,))
        run = simulate(
            state=initial_state,
            action=action,
            response=LinearResponse(epsilon=1.0),
            start_time=0,
            steps=30,
            memory=False,
        )
        assert len(run.states) == len(run.times)
        assert run.initial_state is initial_state
        assert np.allclose(1, run.times[1] - run.times[0])
