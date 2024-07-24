from matplotlib import pyplot as plt
import numpy as np


def make_risk_plots(
    *,
    epsilon_list: list[float | int],
    num_steps: int,
    loss_starts: list[list[float]],
    loss_ends: list[list[float]],
):
    # ------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    offset = 0.8
    # We plot the risk at the beginning and at the end of each round, connecting
    # the two values with a blue line and indicating change in risk due to
    # strategic distribution shift with a dashed green line.
    for idx, epsilon in enumerate(epsilon_list):
        ax = axes[idx // 2][idx % 2]  # type: ignore
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


def make_acc_plots(
    *,
    epsilon_list: list[float | int],
    num_steps: int,
    acc_starts: list[list[float]],
    acc_ends: list[list[float]],
):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    offset = 0.8
    # The performative risk is a surrogate for the underlying metric we care
    # about, accuracy. We can similarly plot accuracy during retraining.
    for idx, epsilon in enumerate(epsilon_list):
        ax = axes[idx // 2][idx % 2]  # type: ignore
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


def make_gap_plots(
    *,
    epsilon_list: list[float | int],
    theta_gaps: list[list[float]],
):
    processed_theta_gaps = [[x for x in tg if 0.0 < x < 0.0] for tg in theta_gaps]
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


def make_feature_weight_plot(
    *,
    epsilon_list: list[float | int],
    thetas: list[list[np.ndarray]],
    changeable_features: list[int],
):
    _, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
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
