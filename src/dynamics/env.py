"""Interactive environment for the credit simulator."""

from typing import Any, SupportsFloat, TypeAlias
from typing_extensions import override

from gymnasium.core import Env, ObsType

from src.conftest import TESTING
from src.dynamics.reward import Reward
from src.dynamics.simulator import Simulator
from src.dynamics.state import State
from src.types import FloatArray

__all__ = ["DynamicEnv"]

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

    @override
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[FloatArray, dict[str, Any]]:
        """Reset the state."""
        self.state = self.initial_state
        self.time = self.start_time
        return super().reset(seed=seed, options=options)

    @override
    def step(self, action: FloatArray) -> StepOutput:
        if not self.action_space.contains(action):
            raise ValueError(f"{action} ({type(action)}) invalid")
        self.time += self.timestep
        # Get the next state from simulation.
        rollout = self.simulator(
            state=self.state,
            action=action,
            steps=self.timestep,
            start_time=self.time,
        )
        self.state = rollout.final_state  # [self.time]
        truncated = terminated = bool(self.time >= self.end_time)
        reward = self.reward_fn.calculate(state=self.state, action=action)
        obs = self.state.asdict()
        info_dict = {}
        return obs, reward, terminated, truncated, info_dict

    @override
    def render(self, mode: str = "human") -> None:
        """Render the environment; unused."""
        del mode


if TESTING:
    import numpy as np

    from src.dynamics.response import LinearResponse
    from src.dynamics.reward import LogisticReward
    from src.loader.credit import CreditData

    def test_env_dynamics() -> None:
        ds = CreditData(seed=0)
        initial_state = State(features=ds.features, labels=ds.labels)
        simulator = Simulator(response=LinearResponse(), memory=False)

        env = DynamicEnv(
            initial_state=initial_state,
            simulator=simulator,
            reward_fn=LogisticReward(),
            start_time=0,
            end_time=1,
        )
        env.reset()
        action1 = np.random.default_rng(0).uniform(
            low=0, high=1, size=(initial_state.num_features,)
        )
        env.step(action=action1)
        action2 = np.random.default_rng(1).uniform(
            low=0, high=1, size=(initial_state.num_features,)
        )
        env.step(action=action2)
