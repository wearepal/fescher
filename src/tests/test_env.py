import numpy as np

from src.dynamics.env import DynamicEnv
from src.dynamics.response import LinearResponse
from src.dynamics.reward import LogisticReward
from src.dynamics.simulator import Simulator
from src.dynamics.state import State
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
    action1 = np.random.default_rng(0).uniform(low=0, high=1, size=(initial_state.num_features,))
    env.step(action=action1)
    action2 = np.random.default_rng(1).uniform(low=0, high=1, size=(initial_state.num_features,))
    env.step(action=action2)
