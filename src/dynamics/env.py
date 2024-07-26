"""Interactive environment for the credit simulator."""

from typing import Any, SupportsFloat, TypeAlias
from typing_extensions import override

from beartype import beartype
from gymnasium.core import Env, ObsType

from src.dynamics.reward import Reward
from src.dynamics.simulator import Simulator
from src.dynamics.state import State, StateDict
from src.types import FloatArray

__all__ = ["DynamicEnv"]

# The gymnasium Environment requires that this be a Tuple
StepOutput: TypeAlias = tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]


class DynamicEnv(Env):
    def __init__(
        self,
        *,
        simulator: Simulator,
        initial_state: State,
        reward_fn: Reward,
        end_time: int,
        start_time: int = 0,
        timestep: int = 1,
    ) -> None:
        self.initial_state = initial_state
        self.state = self.initial_state
        self.action_space = self.state.action_space
        self.observation_space = self.state.observation_space
        self.simulator = simulator
        self.timestep = timestep
        self.start_time = start_time
        self.end_time = end_time
        self.time = self.start_time
        self.reward_fn = reward_fn

    @beartype
    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[StateDict, dict[str, Any]]:
        """Reset the state."""
        super().reset(seed=seed, options=options)
        self.state = self.initial_state
        self.time = self.start_time
        return self.state.asdict(), {} if options is None else options

    @beartype
    @override
    def step(self, action: FloatArray) -> StepOutput:
        if not self.action_space.contains(action):
            raise ValueError(f"{action} ({type(action)}) invalid")
        self.time += self.timestep
        old_features = self.state.features

        # Get the next state from simulation.
        # Do one step of the user adapatation.
        rollout = self.simulator.simulate(
            state=self.state, action=action, steps=self.timestep, start_time=self.time
        )

        self.state = rollout.final_state  # [self.time]
        truncated = terminated = self.time >= self.end_time
        reward = self.reward_fn.calculate(state=self.state, action=action)
        obs = self.state.asdict()
        info_dict = {}

        if not self.simulator.memory:
            self.state.features = old_features

        return obs, reward, terminated, truncated, info_dict

    @beartype
    @override
    def render(self, mode: str = "human") -> None:
        """Render the environment; unused."""
        del mode
