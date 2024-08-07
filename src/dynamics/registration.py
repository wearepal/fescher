"""Environment registration."""

from collections.abc import Callable
from typing import Any, ClassVar, Final, TypeVar, TypedDict, cast, overload
from typing_extensions import Required, Unpack

from beartype import beartype
import gymnasium
from gymnasium.envs.registration import EnvCreator
import numpy as np
from numpy import typing as npt
from ranzen import unwrap_or

from src.dynamics.env import DynamicEnv
from src.dynamics.response import LinearResponse, Response
from src.dynamics.reward import LogisticReward, Reward
from src.dynamics.simulator import Simulator, State
from src.dynamics.state import StateDict
from src.loader.credit import CreditData

__all__ = ["CreditEnvCreator"]

E = TypeVar("E", bound=type[EnvCreator])


@beartype
def make_env(
    *, initial_state: State, epsilon: dict[int, float] | None, memory: bool
) -> gymnasium.Env[StateDict, npt.NDArray[np.float64]]:
    from src.dynamics.registration import CreditEnvCreator

    new_epsilon = unwrap_or(
        epsilon, default={a: 0.0 for a in range(initial_state.features.shape[1])}
    )
    response_fn = LinearResponse(epsilon=new_epsilon)
    env = CreditEnvCreator.as_env(
        initial_state=initial_state, response_fn=response_fn, memory=memory
    )
    env = cast(DynamicEnv, env)
    _ = env.reset()
    return env


@beartype
class RegisterKwargs(TypedDict, total=False):
    id: Required[str]
    reward_threshold: float | None
    nondeterministic: bool
    max_episode_steps: int | None
    order_enforce: bool
    autoreset: bool
    disable_env_checker: bool
    apply_api_compatibility: bool


@overload
def register(fn_: E, /, **kwargs: Unpack[RegisterKwargs]) -> E: ...


@overload
def register(fn_: None = ..., /, **kwargs: Unpack[RegisterKwargs]) -> Callable[[E | None], E]: ...


@beartype
def register(
    fn_: E | None = None, /, **kwargs: Unpack[RegisterKwargs]
) -> E | Callable[[E | None], E | Callable[[E | None], E]]:
    if fn_ is None:

        def _closure(_fn_: E | None, /) -> E | Callable[[E | None], E]:
            return register(_fn_, **kwargs)

        return _closure

    gymnasium.register(
        entry_point=fn_(),
        **kwargs,
    )
    return fn_


@beartype
class CreditEnvKwargs(TypedDict, total=False):
    initial_state: State | None
    response_fn: Response | None
    reward_fn: Reward | None
    memory: bool
    start_time: int
    end_time: int
    timestep: int


_CREDIT_ENV_ID: Final[str] = "Credit-v0"


@beartype
@register(
    id=_CREDIT_ENV_ID,
    nondeterministic=False,
    max_episode_steps=100,
    reward_threshold=0,
)
class CreditEnvCreator:
    ID: ClassVar[str] = _CREDIT_ENV_ID

    def __call__(
        self,
        initial_state: State | None = None,
        response_fn: Response | None = None,
        reward_fn: Reward | None = None,
        memory: bool = False,
        start_time: int = 0,
        end_time: int = 5,
        timestep: int = 1,
        **kwargs: Any,
    ) -> gymnasium.Env[StateDict, npt.NDArray[np.float64]]:
        del kwargs
        initial_state = unwrap_or(initial_state, default=CreditData().as_state())
        reward_fn = unwrap_or(reward_fn, default=LogisticReward(l2_penalty=0.0))
        response_fn = unwrap_or(response_fn, default=LinearResponse(epsilon=None))
        simulator = Simulator(
            response=response_fn,
            memory=memory,
        )

        return DynamicEnv(
            initial_state=initial_state,
            simulator=simulator,
            reward_fn=reward_fn,
            start_time=start_time,
            end_time=end_time,
            timestep=timestep,
        )

    @classmethod
    def as_env(
        cls, **kwargs: Unpack[CreditEnvKwargs]
    ) -> gymnasium.Env[StateDict, npt.NDArray[np.float64]]:
        return gymnasium.make(id=cls.ID, **kwargs)
