"""Interactive environment for the credit simulator."""
from typing import Any
from typing_extensions import override

import gymnasium
from gymnasium.envs.registration import EnvCreator
from ranzen import unwrap_or

from src.conftest import TESTING
from src.env.build import OdeEnv
from src.env.response import LinearResponse, Response
from src.env.reward import LogisticReward, RewardFn
from src.env.simulator import Simulator, State
from src.loader.credit import CreditData


class CreditEnvCreator(EnvCreator):
    @override
    def __call__(
        self,
        initial_state: State | None = None,
        response: Response | None = None,
        reward_fn: RewardFn | None = None,
        memory: bool = False,
        start_time: int = 0,
        end_time: int = 5,
        timestep: int = 1,
        **kwargs: Any,
    ) -> OdeEnv:
        """Construct credit environment."""
        del kwargs
        initial_state = unwrap_or(initial_state, default=CreditData.as_state())
        reward_fn = unwrap_or(reward_fn, default=LogisticReward(l2_penalty=0.0))
        response = unwrap_or(response, default=LinearResponse(epsilon=1.0))
        simulator = Simulator(
            response=response,
            memory=memory,
        )

        return OdeEnv(
            initial_state=initial_state,
            simulator=simulator,
            reward_fn=reward_fn,
            start_time=start_time,
            end_time=end_time,
            timestep=timestep,
        )


def register_envs() -> None:
    gymnasium.register(
        id="Credit-v0",
        entry_point=CreditEnvCreator(),
        nondeterministic=False,
        max_episode_steps=100,
        reward_threshold=0,
    )


register_envs()

if TESTING:

    def test_env_registration() -> None:
        ds = CreditData(seed=0)
        initial_state = State(features=ds.features, labels=ds.labels)
        assert "Credit-v0" in gymnasium.registry
        gymnasium.make("Credit-v0", initial_state=initial_state)
