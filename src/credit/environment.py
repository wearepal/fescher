"""Interactive environment for the credit simulator."""
import copy
from functools import partial
from typing import Any, Protocol, Self, TypeAlias

import numpy as np
import numpy.typing as npt
from credit.agent import LrAgent, NoIntervention
from credit.loader import CreditData
from credit.reward import RewardFn, logistic_loss
from credit.simulator import CreditSTF, Simulator, State
from gym import spaces
from gym.core import ActType, Env, ObsType
from gym.spaces import Space
from typing_extensions import override
from whynot.gym.envs.registration import register


class ObservationFn(Protocol):
    def __call__(self, state: State) -> npt.NDArray:
        ...


StepOutput: TypeAlias = tuple[dict[str, npt.NDArray], float, bool, dict[str, Any]]


class OdeEnv(Env[ObsType, ActType]):
    """Environment builder for simulators derived from dynamical systems."""

    def __init__(
        self,
        *,
        simulator: Simulator,
        action_space: Space[ActType],
        observation_space: Space[ObsType],
        initial_state: State,
        reward_fn: RewardFn,
        end_time: int,
        start_time: int = 0,
        timestep: int = 1,
    ) -> None:
        """Initialize an environment class."""
        self.action_space = action_space
        self.observation_space = observation_space
        self.initial_state = initial_state
        self.state = self.initial_state
        self.simulator = simulator
        self.timestep = timestep
        self.start_time = start_time
        self.end_time = end_time
        self.time = self.start_time
        self.reward_fn = reward_fn
        self.seed()

    @override
    def reset(self) -> None:
        """Reset the state."""
        self.state = self.initial_state
        self.time = self.start_time

    @override
    def step(self, action: npt.NDArray) -> StepOutput:
        """Perform one forward step in the environment."""
        if not self.action_space.contains(action):
            raise ValueError(f"{action} ({type(action)}) invalid")
        self.time += self.timestep
        # Get the next state from simulation.
        rollout = self.simulator(
            state=self.state,
            steps=self.timestep,
            start_time=self.time,
        )
        self.state = rollout[self.time]
        done = bool(self.time >= self.end_time)
        theta = self.simulator.state_transition_fn.agent.theta(self.state.features)
        reward = self.reward_fn(state=self.state, theta=theta)
        obs = self.state.asdict()
        info_dict = {}
        return obs, reward, done, info_dict

    @override
    def render(self, mode="human") -> None:
        """Render the environment, unused."""
        del mode

    def __call__(self) -> Self:
        """Return the class, as if this function were calling the constructor."""
        env = copy.deepcopy(self)
        env.reset()
        return env


# def compute_reward(intervention, state, config):
#     """Compute the reward based on the observed state and choosen intervention."""
#     return logistic_loss(
#         config,
#         state.features,
#         state.labels,
#         theta=theta,
#         intervention.updates["theta"],
#     )


# def compute_intervention(action, time):
#     """Return intervention that changes the classifier parameters to action."""
#     return Intervention(time=time, theta=action)


def credit_action_space(state: State) -> spaces.Box:
    """Return action space for credit simulator.

    The action space is the vector of possible logistic regression
    parameters, which depends on the dimensionality of the features.
    """
    num_features = state.features.shape[-1]
    return spaces.Box(low=-np.inf, high=np.inf, shape=(num_features,), dtype=np.float64)


def credit_observation_space(initial_state) -> spaces.Dict:
    """Return observation space for credit simulator.

    The observation space is the vector of possible datasets, which
    must have the same dimensions as the initial state.
    """
    return spaces.Dict(
        {
            "features": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=initial_state.features.shape,
                dtype=np.float64,
            ),
            "labels": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=initial_state.labels.shape,
                dtype=np.float64,
            ),
        }
    )


def build_credit_env(initial_state=None) -> OdeEnv:
    """Construct credit environment that is parameterized by the initial state.

    This allows the user to specify different datasets other than the default
    Credit dataset.
    """
    if initial_state is None:
        initial_state = CreditData.as_state()
    elif not isinstance(initial_state, State):
        raise ValueError(f"Initial state must be an instance of '{State.__name__}'")

    stf = CreditSTF(
        agent=LrAgent(),
        intervention=NoIntervention(),
    )
    simulator = Simulator(
        state_transition_fn=stf,
    )

    return OdeEnv(
        initial_state=initial_state,
        simulator=simulator,
        action_space=credit_action_space(initial_state),
        observation_space=credit_observation_space(initial_state),
        reward_fn=partial(logistic_loss, weight_decay=0.0),
        start_time=0,
        end_time=5,
        timestep=1,
    )


register(
    id="Credit-v0",
    entry_point=build_credit_env,
    max_episode_steps=100,
    reward_threshold=0,
)
