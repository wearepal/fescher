from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import sys

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent / ".."))


from hydra_zen import ZenStore, zen

from src.dynamics.env import DynamicEnv
from src.dynamics.response import LinearResponse
from src.dynamics.state import State
from src.loader.credit import CreditData, Data
from src.models.lr import Lr, logistic_loss


@dataclass(unsafe_hash=True, kw_only=True)
class EpisodeRecord:
    loss_start: list[float] = field(default_factory=list)
    loss_end: list[float] = field(default_factory=list)
    acc_start: list[float] = field(default_factory=list)
    acc_end: list[float] = field(default_factory=list)
    theta_gap: list[float] = field(default_factory=list)
    theta: list[np.ndarray] = field(default_factory=list)


def repeated_risk_minimization(
    *, env: DynamicEnv, num_steps: int, lr: Lr, l2_penalty: float
) -> EpisodeRecord:
    """Run repeated risk minimization for num_iters steps"""
    # Track loss and accuracy before/after updating model on new distribution
    record = EpisodeRecord()
    # Warm-start with baseline classifier
    theta = np.copy(lr.weights)
    for _ in range(num_steps):
        # Deploy classifier and observe strategic response
        observation = env.step(theta)[0]
        features_strat, labels = observation["features"], observation["labels"]

        # Evaluate loss and accuracy on the new distribution
        record.loss_start.append(
            logistic_loss(
                x=features_strat,
                y=labels,
                weights=theta,
                l2_penalty=l2_penalty,
            )
        )
        record.acc_start.append(lr.acc(features=features_strat, labels=labels))
        # Learn a new model on the induced distribution bootstrapped from the previous model.
        lr = lr.fit(x=features_strat, y=labels)
        # Evaluate loss and accuracy on the strategic distribution after training
        record.loss_end.append(
            lr.loss(
                x=features_strat,
                y=labels,
            )
        )
        record.acc_end.append(lr.acc(features=features_strat, labels=labels))
        # Track distance (in terms of Euclidean norm) between iterates
        record.theta_gap.append(np.linalg.norm(lr.weights - theta))  # type: ignore
        record.theta.append(lr.weights)
        theta = np.copy(lr.weights)
    return record


def make_env(*, initial_state: State, epsilon: float) -> DynamicEnv:
    from src.dynamics.registration import CreditEnvCreator

    # response_fn = LinearResponse(epsilon=epsilon, changeable_features=[0, 5, 7])
    response_fn = LinearResponse(epsilon=epsilon, changeable_features=[2, 6, 8])
    env = CreditEnvCreator.as_env(
        initial_state=initial_state,
        response_fn=response_fn,
    )
    env.reset()
    return env


store = ZenStore()
data_store = store(group="dataset")
data_store(CreditData, seed=0, name="credit")


@store(name="fescher", hydra_defaults=["_self_", {"dataset": "credit"}])
def main(dataset: Data):
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
    l2_penalty = 1.0 / num_agents

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

    for epsilon_idx, epsilon in tqdm(enumerate(epsilon_list)):
        lr = Lr(l2_penalty=l2_penalty).fit(
            x=base_x,
            y=base_y,
        )
        baseline_acc = lr.acc(features=base_x, labels=base_y)
        logger.info(f"Baseline logistic regresion model accuracy: {100 * baseline_acc:.2f}%")
        env = make_env(initial_state=initial_state, epsilon=epsilon)
        logger.info(f"Running retraining for epsilon {epsilon:.2f}")
        record = repeated_risk_minimization(
            env=env, lr=lr, num_steps=num_steps, l2_penalty=l2_penalty
        )
        loss_starts.append(record.loss_start)
        loss_ends.append(record.loss_end)
        acc_starts.append(record.acc_start)
        acc_ends.append(record.acc_end)
        theta_gaps.append(record.theta_gap)
        thetas.append(record.theta)

    changeable_features = env.simulator.response.changeable_features  # type: ignore
    # ------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    # We plot the risk at the beginning and at the end of each round, connecting
    # the two values with a blue line and indicating change in risk due to
    # strategic distribution shift with a dashed green line.
    for idx, epsilon in enumerate(epsilon_list):
        ax = axes[idx // 2][idx % 2]  # type: ignore
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
        ax = axes[idx // 2][idx % 2]  # type: ignore
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
        # ax.set_ylim([0.5, 0.75])
    plt.subplots_adjust(hspace=0.25)
    plt.show()

    processed_theta_gaps = [[x for x in tg if x != 0.0] for tg in theta_gaps]
    _, ax = plt.subplots(figsize=(10, 8))

    # Finally, we plot the distance between consecutive iterates. This is the
    # quantity bounded by the theorems in Performative Prediction, which shows
    # that repeated risk minimization converges in domain to a stable point.
    for idx, (gaps, eps) in enumerate(zip(processed_theta_gaps, epsilon_list)):
        label = f"$\\varepsilon$ = {eps}"
        if idx == 0:
            ax.semilogy(
                gaps,
                label=label,
                linewidth=3,
                alpha=1,
                # markevery=[-1],
                marker="*",
                linestyle=(0, (1, 1)),
            )
        elif idx == 1:
            ax.semilogy(
                gaps,
                label=label,
                linewidth=3,
                alpha=1,
                # markevery=[-1],
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

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    # The performative risk is a surrogate for the underlying metric we care
    # about, accuracy. We can similarly plot accuracy during retraining.
    for idx, epsilon in enumerate(epsilon_list):
        ax = axes[idx // 2][idx % 2]  # type: ignore
        ax.set_title(rf"Feature importance, $\epsilon$={epsilon}")
        theta = np.stack(thetas[idx], axis=1)
        for i in range(theta.shape[0] - 1):
            if i in changeable_features:
                ax.plot(theta[i], label=f"{i}*")
            if i not in changeable_features:
                ax.plot(theta[i], label=f"{i}", linestyle="dashed")

        ax.set_xlabel("Step")
        ax.set_ylabel("Weight")
        ax.legend(loc="center right")
        # ax.set_ylim([0.5, 0.75])
    plt.subplots_adjust(hspace=0.25)
    plt.show()


if __name__ == "__main__":
    store.add_to_hydra_store()
    zen(main).hydra_main(
        config_name="fescher",
        version_base="1.1",
        config_path=None,
    )
