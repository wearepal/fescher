import copy
from dataclasses import dataclass, fields
from typing import Generic, Protocol, TypeVar

import numpy as np
import numpy.typing as npt
from typing_extensions import TypeAlias, override

from src.conftest import TESTING
from src.credit.agent import Agent, Intervention, LrAgent, NoIntervention

Features: TypeAlias = npt.NDArray[np.floating]
Labels: TypeAlias = npt.NDArray[np.floating]

__all__ = ["State", "simulate", "Rollout", "Simulator", "StateTransitionFn"]


@dataclass(kw_only=True)
class State:
    """Simulative state."""

    features: Features
    labels: Labels

    def __len__(self) -> int:
        return len(fields(self))

    def asdict(self) -> dict[str, npt.NDArray[np.floating]]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(kw_only=True)
class Rollout:
    r"""Encapsulate a trajectory from a dynamical system simulator.

    :states: Sequence of states :math:`x_{t_1}, x_{t_2}, \dots` produced by the simulator.
    :times: Sequence of sampled times :math:`{t_1},{t_2}, \dots` at which the states were recorded.
    """

    states: list[State]
    times: list[int]

    def __post_init__(self) -> None:
        """Error checking called after the constructor."""
        if len(self.states) != len(self.times):
            msg = "Input states and times must be the same length!  {} \neq {}"
            raise ValueError(msg.format(len(self.states), len(self.times)))

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


class StateTransitionFn(Protocol):
    def __call__(self, *, state: State, time: int) -> State:
        ...


A = TypeVar("A", bound=Agent)


@dataclass(kw_only=True)
class CreditSTF(StateTransitionFn, Generic[A]):
    agent: A
    intervention: Intervention[A]

    @override
    def __call__(self, *, state: State, time: int) -> State:
        agent = self.intervention(time=time, agent=self.agent)
        features = agent.update(features=state.features)
        state = State(features=features, labels=state.features)
        return state


def simulate(
    *,
    state: State,
    steps: int,
    state_transition_fn: StateTransitionFn,
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
        state = state_transition_fn(state=state, time=time)
        states.append(state)
        times.append(time + 1)

    return Rollout(states=states, times=times)


STF = TypeVar("STF", bound=StateTransitionFn)


@dataclass(kw_only=True)
class Simulator(Generic[STF]):
    state_transition_fn: STF
    memory: bool = False

    def __call__(self, *, state: State, steps: int, start_time: int = 0) -> Rollout:
        return simulate(
            state=state,
            steps=steps,
            state_transition_fn=self.state_transition_fn,
            memory=self.memory,
            start_time=start_time,
        )


if TESTING:

    def test_dynamics_initial_state() -> None:
        """For ODE simulators, ensure the iniitial_state is returned by reference in run."""
        from src.credit.loader import CreditData

        ds = CreditData(seed=0)
        initial_state = State(features=ds.features, labels=ds.labels)
        stf = CreditSTF(
            agent=LrAgent(),
            intervention=NoIntervention(),
        )
        run = simulate(
            state=initial_state,
            state_transition_fn=stf,
            start_time=0,
            steps=30,
            memory=False,
        )
        assert len(run.states) == len(run.times)
        assert run.initial_state is initial_state
        assert np.allclose(1, run.times[1] - run.times[0])
