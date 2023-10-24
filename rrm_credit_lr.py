from __future__ import annotations
from dataclasses import dataclass, field
from typing import Generic, TypeAlias, TypeVar, Union, cast

from loguru import logger
import lr
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from tqdm import tqdm
import whynot.gym as gym
from whynot.simulators.credit import Config, CreditData, State

FloatArray: TypeAlias = npt.NDArray[np.floating]
X = TypeVar("X", bound=Union[pl.Series, pl.DataFrame, FloatArray])
Y = TypeVar("Y", bound=Union[pl.Series, pl.DataFrame, FloatArray])


@dataclass(unsafe_hash=True, kw_only=True)
class Dataset(Generic[X, Y]):
    x: X
    y: Y

    def to_numpy(self) -> Dataset[FloatArray, FloatArray]:
        x = self.x
        y = self.y
        if not isinstance(x, np.ndarray):
            x = x.to_numpy()
        if not isinstance(y, np.ndarray):
            y = y.to_numpy()
        return Dataset(x=x, y=y)


@dataclass(unsafe_hash=True, kw_only=True)
class EpisodeRecord:
    loss_start: list[float] = field(default_factory=list)
    loss_end: list[float] = field(default_factory=list)
    acc_start: list[float] = field(default_factory=list)
    acc_end: list[float] = field(default_factory=list)
    theta_gap: list[float] = field(default_factory=list)


def repeated_risk_minimization(
    *,
    env: gym.Env,
    base_theta: FloatArray,
    epsilon: float,
    num_steps: int,
) -> EpisodeRecord:
    """Run repeated risk minimization for num_iters steps"""
    config = cast(Config, env.config)  # type: ignore
    config.epsilon = epsilon  # type: ignore
    config.l2_penalty = l2_penalty  # type: ignore
    env.reset()

    # Track loss and accuracy before/after updating model on new distribution
    record = EpisodeRecord()
    # Warm-start with baseline classifier
    theta = np.copy(base_theta)
    for _ in range(num_steps):
        # Deploy classifier and observe strategic response
        observation, _, _, _ = env.step(theta)
        features_strat, labels = observation["features"], observation["labels"]

        # Evaluate loss and accuracy on the new distribution
        record.loss_start.append(
            lr.logistic_loss(x=features_strat, y=labels, weights=theta, l2_penalty=l2_penalty)
        )
        record.acc_start.append(((features_strat.dot(theta) > 0) == labels).mean())

        # Learn a new model on the induced distribution
        theta_new = lr.fit_lr_with_gd(
            x=features_strat,
            y=labels,
            l2_penalty=l2_penalty,
            theta_init=np.copy(theta),
        )

        # Evaluate loss and accuracy on the strategic distribution after training
        record.loss_end.append(
            lr.logistic_loss(
                x=features_strat,
                y=labels,
                weights=theta_new,
                l2_penalty=l2_penalty,
            )
        )
        record.acc_end.append(((features_strat.dot(theta_new) > 0) == labels).mean())

        # Track distance (in terms of Euclidean norm) between iterates
        record.theta_gap.append(np.linalg.norm(theta_new - theta))

        theta = np.copy(theta_new)

    return record


if __name__ == "__main__":
    # We use the credit simulator, which is a strategic classification
    # simulator based on the 'Kaggle Give Me Some Credit' (GMSC) dataset.
    ds = CreditData
    # The state of the environment is a dataset consisting of (1) financial
    # features of individuals, e.g. DebtRatio, and (2) a binary label indicating
    # whether an individual experienced financial distress in the subsequent two
    # years.
    initial_state = State(features=ds.features, labels=ds.labels)
    env: gym.Env = gym.make("Credit-v0", initial_state=initial_state)
    env.seed(1)
    features, labels = env.reset()
    ds = initial_state.values()
    base_x, base_y = ds["features"], ds["labels"]
    num_agents, num_features = base_x.shape
    logger.info(f"The dataset is made up of {num_agents} agents and {num_features} features.")

    # Fit a rudimentary LR model to the data.
    l2_penalty = 1.0 / num_agents
    baseline_theta = lr.fit_lr_with_gd(
        x=base_x,
        y=base_y,
        l2_penalty=l2_penalty,
    )
    baseline_acc = (((base_x @ baseline_theta) > 0) == base_y).mean()
    logger.info(f"Baseline logistic regresion model accuracy: {100 * baseline_acc:.2f}%")

    epsilon_list = [
        1,
        80,
        150,
        1000,
    ]
    num_steps = 25

    loss_starts, acc_starts, loss_ends, acc_ends, theta_gaps = [], [], [], [], []

    for epsilon_idx, epsilon in tqdm(enumerate(epsilon_list)):
        print(f"Running retraining for epsilon {epsilon:.2f}")
        record = repeated_risk_minimization(
            env=env,
            base_theta=baseline_theta,
            epsilon=epsilon,
            num_steps=num_steps,
        )
        loss_starts.append(record.loss_start)
        loss_ends.append(record.loss_end)
        acc_starts.append(record.acc_start)
        acc_ends.append(record.acc_end)
        theta_gaps.append(record.theta_gap)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

    # We plot the risk at the beginning and at the end of each round, connecting
    # the two values with a blue line and indicating change in risk due to
    # strategic distribution shift with a dashed green line.
    for idx, epsilon in enumerate(epsilon_list):
        ax = axes[idx // 2][idx % 2]
        offset = 0.8
        ax.set_title(rf"Performative Risk, $\epsilon$={epsilon}", fontsize=20)
        for i in range(2, num_steps):
            ax.plot([i, i + offset], [loss_starts[idx][i], loss_ends[idx][i]], "b*-")
            if i < num_steps - 1:
                ax.plot(
                    [i + offset, i + 1],
                    [loss_ends[idx][i], loss_starts[idx][i + 1]],
                    "g--",
                )

        ax.set_xlabel("Step", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        ax.set_yscale("log")

    plt.subplots_adjust(hspace=0.25)
    plt.show()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    # The performative risk is a surrogate for the underlying metric we care
    # about, accuracy. We can similarly plot accuracy during retraining.
    for idx, epsilon in enumerate(epsilon_list):
        ax = axes[idx // 2][idx % 2]
        offset = 0.8
        ax.set_title(rf"Performative Accuracy, $\epsilon$={epsilon}", fontsize=20)
        for i in range(2, num_steps):
            ax.plot([i, i + offset], [acc_starts[idx][i], acc_ends[idx][i]], "b*-")
            if i < num_steps - 1:
                ax.plot(
                    [i + offset, i + 1],
                    [acc_ends[idx][i], acc_starts[idx][i + 1]],
                    "g--",
                )

        ax.set_xlabel("Step", fontsize=16)
        ax.set_ylabel("Accuracy", fontsize=16)
        ax.set_ylim([0.5, 0.75])
    plt.subplots_adjust(hspace=0.25)
    plt.show()

    processed_theta_gaps = [[x for x in tg if x != 0.0] for tg in theta_gaps]
    _, ax = plt.subplots(figsize=(10, 8))

    # Finally, we plot the distance between consecutive iterates. This is the
    # quantity bounded by the theorems in Performative Prediction, which shows
    # that repeated risk minimization converges in domain to a stable point.
    for idx, (gaps, eps) in enumerate(zip(processed_theta_gaps, epsilon_list)):
        label = "$\\varepsilon$ = {}".format(eps)
        if idx == 0:
            ax.semilogy(
                gaps,
                label=label,
                linewidth=3,
                alpha=1,
                markevery=[-1],
                marker="*",
                linestyle=(0, (1, 1)),
            )
        elif idx == 1:
            ax.semilogy(
                gaps,
                label=label,
                linewidth=3,
                alpha=1,
                markevery=[-1],
                marker="*",
                linestyle="solid",
            )
        else:
            ax.semilogy(gaps, label=label, linewidth=3)

    ax.set_title("Convergence in Domain for Repeated Risk Minimization", fontsize=18)
    ax.set_xlabel("Iteration $t$", fontsize=18)
    ax.set_ylabel(r"Distance Between Iterates: $\|\theta_{t+1} - \theta_{t}\|_2 $", fontsize=14)
    ax.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    plt.show()
