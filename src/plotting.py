from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt


def make_risk_plots(
    *,
    epsilon: dict[int, float],
    num_steps: int,
    loss_starts: list[float],
    loss_ends: list[float],
    out_dir: Path,
):
    # ------------------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------------------
    _, ax = plt.subplots(figsize=(16, 12))
    offset = 0.8
    # We plot the risk at the beginning and at the end of each round, connecting
    # the two values with a blue line and indicating change in risk due to
    # strategic distribution shift with a dashed green line.

    ax.set_title(rf"Performative Risk, $\epsilon$={epsilon}", fontsize=20)
    for i in range(2, num_steps):
        ax.plot([i, i + offset], [loss_starts[i], loss_ends[i]], "b*-")
        if i < num_steps - 1:
            ax.plot(
                [i + offset, i + 1],
                [loss_ends[i], loss_starts[i + 1]],
                "g--",
            )

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_yscale("log")

    plt.subplots_adjust(hspace=0.25)

    # Save the plot
    plot_path = out_dir / "risk_plot.png"
    plt.savefig(plot_path)
    plt.close()


def make_acc_plots(
    *,
    epsilon: dict[int, float],
    num_steps: int,
    acc_starts: list[float],
    acc_ends: list[float],
    out_dir: Path,
):
    _, ax = plt.subplots(figsize=(16, 12))
    offset = 0.8
    # The performative risk is a surrogate for the underlying metric we care
    # about, accuracy. We can similarly plot accuracy during retraining.

    ax.set_title(rf"Performative Accuracy, $\epsilon$={epsilon}", fontsize=20)
    for i in range(2, num_steps):
        ax.plot([i, i + offset], [acc_starts[i], acc_ends[i]], "b*-")
        if i < num_steps - 1:
            ax.plot(
                [i + offset, i + 1],
                [acc_ends[i], acc_starts[i + 1]],
                "g--",
            )

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Accuracy", fontsize=16)
    # ax.set_ylim([0.5, 0.75])
    plt.subplots_adjust(hspace=0.25)
    # Save the plot
    plot_path = out_dir / "acc_plot.png"
    plt.savefig(plot_path)
    plt.close()


def make_gap_plots(
    *,
    epsilon: dict[int, float],
    theta_gaps: list[float],
    out_dir: Path,
):
    processed_theta_gaps = [x for x in theta_gaps if x > 0.0 or x < 0.0]
    _, ax = plt.subplots(figsize=(10, 8))

    # Finally, we plot the distance between consecutive iterates. This is the
    # quantity bounded by the theorems in Performative Prediction, which shows
    # that repeated risk minimization converges in domain to a stable point.
    label = f"$\\varepsilon$ = {epsilon}"
    ax.semilogy(processed_theta_gaps, label=label, linewidth=3)
    ax.set_title("Convergence in Domain for Repeated Risk Minimization", fontsize=18)
    ax.set_xlabel("Iteration $t$", fontsize=18)
    ax.set_ylabel(r"Distance Between Iterates: $\|\theta_{t+1} - \theta_{t}\|_2 $", fontsize=14)
    ax.tick_params(labelsize=18)
    plt.legend(fontsize=18)
    # Save the plot
    plot_path = out_dir / "gap_plot.png"
    plt.savefig(plot_path)
    plt.close()


def make_feature_weight_plot(
    *,
    epsilon: dict[int, float],
    thetas: list[npt.NDArray[np.float64]],
    show_bias: bool,
    out_dir: Path,
):
    _, ax = plt.subplots(figsize=(12, 8))
    # The performative risk is a surrogate for the underlying metric we care
    # about, accuracy. We can similarly plot accuracy during retraining.
    include_bias = 0 if show_bias else 1

    ax.set_title(rf"Feature importance, $\epsilon$={epsilon}")
    theta = np.stack(thetas, axis=1)
    for i in range(theta.shape[0] - include_bias):
        if i in epsilon and epsilon[i] > 0:
            ax.plot(abs(theta[i]), label=f"{i}", linestyle="dashed")
        else:
            ax.plot(abs(theta[i]), label=f"{i}*")

    ax.set_xlabel("Step")
    ax.set_ylabel("Weight")
    ax.legend(loc="center right")
    # ax.set_ylim([0.5, 0.75])
    plt.subplots_adjust(hspace=0.25)
    # Save the plot
    plot_path = out_dir / "feature_weight_plot.png"
    plt.savefig(plot_path)
    plt.close()
