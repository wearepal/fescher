from __future__ import annotations
from dataclasses import dataclass

from hydra_zen import ZenStore, zen
from loguru import logger
import numpy as np
from tqdm import tqdm

from src.dynamics.registration import make_env
from src.loader.credit import CreditData, Data
from src.models.lr import Lr
from src.plotting import make_acc_plots, make_feature_weight_plot, make_gap_plots, make_risk_plots
from src.repeated_risk_min import repeated_risk_minimization


@dataclass
class ExperimentSettings:
    l2_penalty: float


store = ZenStore()
data_store = store(group="dataset")
data_store(CreditData, seed=0, name="credit")

exp_store = store(group="experiment")
exp_store(ExperimentSettings, l2_penalty=1e-4, name="base")


@store(name="fescher", hydra_defaults=["_self_", {"dataset": "credit"}, {"experiment": "base"}])
def main(dataset: Data, experiment: ExperimentSettings):
    # We use the credit simulator, which is a strategic classification
    # simulator based on the 'Kaggle Give Me Some Credit' (GMSC) dataset.
    initial_state = dataset.as_state()
    # The state of the environment is a dataset consisting of (1) financial
    # features of individuals, e.g. DebtRatio, and (2) a binary label indicating
    # whether an individual experienced financial distress in the subsequent two
    # years.
    base_x, base_y = initial_state.features, initial_state.labels
    num_agents, num_features = base_x.shape
    logger.info(f"The dataset is made up of {num_agents} agents and {num_features} features.")

    # Fit a rudimentary LR model to the data.
    # l2_penalty = 1.0 / num_agents # Commented this out, but not deleting yet

    epsilon_list = [
        # 0.001,
        # 0.01,
        # 0.1,
        # 0.2,
        # 0.3,
        # 0.4,
        # 0.5,
        0.6,
        1,
        2,
        3,
        # 10,
        # 80,
        # 150,
        # 1000,
    ]
    num_steps = 1000

    loss_starts: list[list[float]] = []
    acc_starts: list[list[float]] = []
    loss_ends: list[list[float]] = []
    acc_ends: list[list[float]] = []
    theta_gaps: list[list[float]] = []
    thetas: list[list[np.ndarray]] = []

    for epsilon in tqdm(epsilon_list):
        lr = Lr(l2_penalty=experiment.l2_penalty).fit(
            x=base_x,
            y=base_y,
        )
        baseline_acc = lr.acc(features=base_x, labels=base_y)
        logger.info(f"Baseline logistic regresion model accuracy: {100 * baseline_acc:.2f}%")
        env = make_env(initial_state=initial_state, epsilon=epsilon)
        logger.info(f"Running retraining for epsilon {epsilon:.2f}")
        record = repeated_risk_minimization(
            env=env, lr=lr, num_steps=num_steps, l2_penalty=experiment.l2_penalty
        )
        loss_starts.append(record.loss_start)
        loss_ends.append(record.loss_end)
        acc_starts.append(record.acc_start)
        acc_ends.append(record.acc_end)
        theta_gaps.append(record.theta_gap)
        thetas.append(record.theta)

    changeable_features = env.simulator.response.changeable_features  # type: ignore
    make_risk_plots(
        epsilon_list=epsilon_list, num_steps=num_steps, loss_starts=loss_starts, loss_ends=loss_ends
    )
    make_acc_plots(
        epsilon_list=epsilon_list, num_steps=num_steps, acc_starts=acc_starts, acc_ends=acc_ends
    )
    make_gap_plots(epsilon_list=epsilon_list, theta_gaps=theta_gaps)
    make_feature_weight_plot(
        epsilon_list=epsilon_list, thetas=thetas, changeable_features=changeable_features
    )


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="fescher",
        version_base="1.1",
        config_path=None,
    )
