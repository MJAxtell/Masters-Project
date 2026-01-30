"""AI Use-Disclaimer. Generative AI, specifically, ChatGPT, has been used to help with some syntax
in quickly constructing MatplotLib plots. All code is handwritten and reviewed and all data manually collected
from the training processes."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

def plot_metric(
    filename,
    ylabel,
    title,
    inset_xlim,
    inset_ylim,
    inset_xticks,
    inset_yticks,
    output_file
):
    """ Setup """
    existential_inclusive = pd.read_csv(filename, sep="\t")
    x = existential_inclusive["Test Cycle"]
    configs = ["A", "B", "C"]

    """
    ---------------------------------
    Metric Plot
    ---------------------------------
    """
    plt.figure(figsize=(7.5, 4.5))

    for label in configs:
        y = existential_inclusive[label]

        plt.plot(
            x, y,
            marker='o',
            linewidth=1.4,
            markersize=5,
            alpha=0.75,
            label=label
        )

    plt.ylim(80, 101.5)
    plt.xticks(range(1, 21))

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        title="Configuration",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True
    )

    """Sub-plot for outliers"""

    ax = plt.gca()
    axins = inset_axes(
        ax,
        width="35%",
        height="35%",
        loc="lower left",
        bbox_to_anchor=(1.08, 0.1, 1, 1),
        bbox_transform=ax.transAxes,
        borderpad=0
    )

    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


    for label in configs:
        axins.plot(
            x, existential_inclusive[label],
            marker="o",
            linewidth=1,
            markersize=3,
            alpha=0.75
        )

    axins.set_ylim(*inset_ylim)
    axins.set_xlim(*inset_xlim)
    axins.set_title("Outliers", fontsize=8)

    axins.set_xticks(inset_xticks)
    axins.set_yticks(inset_yticks)
    axins.tick_params(axis="both", which="major", labelsize=8)

    axins.grid(True, alpha=0.4)
    for spine in axins.spines.values():
        spine.set_linewidth(0.8)

    """
    ---------------------------------
    Show Figures
    ---------------------------------
    """

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_metric_second(
    filename,
    ylabel,
    title,
    inset_xlim,
    inset_ylim,
    inset_xticks,
    inset_yticks,
    output_file,
    subgraph=True,
):
    """ Setup """
    existential_inclusive = pd.read_csv(filename, sep="\t")
    x = existential_inclusive["Test Cycle"]
    configs = ["D", "E", "F"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    start_idx = 3
    rotated_colors = colors[start_idx:] + colors[:start_idx]

    """
    ---------------------------------
    Metric Plot
    ---------------------------------
    """
    plt.figure(figsize=(7.5, 4.5))

    for label, color in zip(configs, rotated_colors):
        y = existential_inclusive[label]

        plt.plot(
            x, y,
            marker='o',
            linewidth=1.4,
            markersize=5,
            alpha=0.75,
            color=color,
            label=label
        )

    plt.ylim(80, 101.5)
    plt.xticks(range(1, 21))

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        title="Configuration",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True
    )

    """Sub-plot for outliers"""

    if subgraph:
        ax = plt.gca()
        axins = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc="lower left",
            bbox_to_anchor=(1.08, 0.1, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        ax.grid(True, alpha=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)


        for label in configs:
            axins.plot(
                x, existential_inclusive[label],
                marker="o",
                linewidth=1,
                markersize=3,
                alpha=0.75
            )

        axins.set_ylim(*inset_ylim)
        axins.set_xlim(*inset_xlim)
        axins.set_title("Outliers", fontsize=8)

        axins.set_xticks(inset_xticks)
        axins.set_yticks(inset_yticks)
        axins.tick_params(axis="both", which="major", labelsize=8)

        axins.grid(True, alpha=0.4)
        for spine in axins.spines.values():
            spine.set_linewidth(0.8)

    """
    ---------------------------------
    Show Figures
    ---------------------------------
    """

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_metric_full(
    filename,
    ylabel,
    title,
    output_file
):
    """ Setup """
    existential_inclusive = pd.read_csv(filename, sep="\t")
    x = existential_inclusive["Test Cycle"]
    configs = ["A", "B", "C"]

    """
    ---------------------------------
    Metric Plot (Full Range, No Inset)
    ---------------------------------
    """
    plt.figure(figsize=(7.5, 4.5))

    for label in configs:
        y = existential_inclusive[label]

        plt.plot(
            x, y,
            marker='o',
            linewidth=1.4,
            markersize=5,
            alpha=0.75,
            label=label
        )

    plt.ylim(0, 102)
    plt.xticks(range(1, 21))

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        title="Configuration",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True
    )

    ax = plt.gca()

    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    """
    ---------------------------------
    Show Figures
    ---------------------------------
    """

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_metric_full_second(
    filename,
    ylabel,
    title,
    output_file
):
    """ Setup """
    existential_inclusive = pd.read_csv(filename, sep="\t")
    x = existential_inclusive["Test Cycle"]
    configs = ["D", "E", "F"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    start_idx = 3
    rotated_colors = colors[start_idx:] + colors[:start_idx]

    """
    ---------------------------------
    Metric Plot (Full Range, No Inset)
    ---------------------------------
    """
    plt.figure(figsize=(7.5, 4.5))

    for label, color in zip(configs, rotated_colors):
        y = existential_inclusive[label]

        plt.plot(
            x, y,
            marker='o',
            linewidth=1.4,
            markersize=5,
            alpha=0.75,
            color=color,
            label=label
        )

    plt.ylim(0, 102)
    plt.xticks(range(1, 21))

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(
        title="Configuration",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True
    )

    ax = plt.gca()

    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    """
    ---------------------------------
    Show Figures
    ---------------------------------
    """

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_rule_counts(
    filename,
    ylabel,
    title,
    output_file
):
    """ Setup """
    df = pd.read_csv(filename, sep="\t")
    x = df["Test Cycle"]
    configs = ["A", "B", "C"]

    """
    ---------------------------------
    Rule Count Plot (Proposed vs Environment)
    ---------------------------------
    """
    plt.figure(figsize=(7.5, 4.5))

    for label in configs:
        proposed = df[f"{label}_proposed"]
        env = df[f"{label}_env"]

        line, = plt.plot(
            x, proposed,
            marker='o',
            linewidth=1.4,
            markersize=5,
            alpha=0.8,
            label=f"{label} proposed"
        )

        color = line.get_color()

        plt.plot(
            x, env,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            color=color,
            label=f"{label} environment"
        )

    plt.xticks(range(1, 21))

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)

    ax = plt.gca()

    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        title="Configuration"
    )

    """
    ---------------------------------
    Show Figures
    ---------------------------------
    """

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_rule_counts_second(
    filename,
    ylabel,
    title,
    output_file
):
    """ Setup """
    df = pd.read_csv(filename, sep="\t")
    x = df["Test Cycle"]
    configs = ["D", "E", "F"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    start_idx = 3  # ← choose where to start
    rotated_colors = colors[start_idx:] + colors[:start_idx]

    """
    ---------------------------------
    Rule Count Plot (Proposed vs Environment)
    ---------------------------------
    """
    plt.figure(figsize=(7.5, 4.5))

    for label, color in zip(configs, rotated_colors):
        proposed = df[f"{label}_proposed"]
        env = df[f"{label}_env"]

        line, = plt.plot(
            x, proposed,
            marker='o',
            linewidth=1.4,
            markersize=5,
            alpha=0.8,
            color=color,
            label=f"{label} proposed"
        )

        color = line.get_color()

        plt.plot(
            x, env,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            color=color,
            label=f"{label} environment"
        )

    plt.xticks(range(1, 21))

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)

    ax = plt.gca()

    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
        title="Configuration"
    )

    """
    ---------------------------------
    Show Figures
    ---------------------------------
    """

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_overall_percentages(
    filename,
    ylabel,
    title,
    output_file
):
    # -----------------
    # Load data
    # -----------------
    df = pd.read_csv(filename, sep="\t")

    metrics = df["metric"].values
    configs = ["A", "B", "C", "D", "E", "F"]

    num_metrics = len(metrics)
    x = np.arange(num_metrics)

    # -----------------
    # Colour mapping (metric → colour)
    # -----------------
    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = (base_colors * ((num_metrics // len(base_colors)) + 1))[:num_metrics]

    legend_handles = [
        Patch(facecolor=colors[i], label=metrics[i])
        for i in range(num_metrics)
    ]

    # -----------------
    # Figure + GridSpec
    # 3 columns for plots, 1 for legend
    # -----------------
    fig = plt.figure(figsize=(14.5, 6.5))
    gs = GridSpec(
        2, 4,
        figure=fig,
        width_ratios=[1, 1, 1, 0.75],  # last column = legend space
        wspace=0.25,
        hspace=0.25
    )

    axes = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]

    for ax, cfg in zip(axes, configs):
        bars = ax.bar(x, df[cfg], color=colors)
        for rect, value in zip(bars, df[cfg]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7
            )

        ax.set_title(f"Configuration {cfg}")
        ax.set_ylim(0, 105)
        ax.set_xticks([])

        ax.grid(axis="y", alpha=0.4)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)

    # -----------------
    # Legend axis (EMPTY AXIS)
    # -----------------
    legend_ax = fig.add_subplot(gs[:, 3])
    legend_ax.axis("off")

    legend_ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.75, 0.75),
        frameon=True,
        title="Metric"
    )

    # -----------------
    # Global labels
    # -----------------
    fig.suptitle(title, fontsize=13)
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")

    # -----------------
    # Save
    # -----------------
    plt.savefig(output_file)
    plt.close(fig)

def plot_overall_counts(
    filename,
    ylabel,
    title,
    output_file
):
    """ Setup """
    df = pd.read_csv(filename, sep="\t")

    metric_name = df.iloc[0, 0]
    configs = df.columns[1:]
    values = df.iloc[0, 1:].astype(float)

    plt.figure(figsize=(6.5, 4.5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    bar_colors = colors[:len(values)]

    bars = plt.bar(
        configs,
        values,
        color=bar_colors,
        alpha=0.85
    )

    plt.xlabel("Configuration")
    plt.ylabel(ylabel)
    plt.title(title if title is not None else metric_name)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    ax = plt.gca()
    ax.grid(True, axis="y", alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()

"""Existential Inclusive Accuracy"""
plot_metric(
    filename="existential_inclusive.tsv",
    ylabel="Inclusive existential accuracy (%)",
    title="Inclusive Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    inset_xlim=(8.5, 13.5),
    inset_ylim=(40, 90),
    inset_xticks=[9, 10, 11, 12, 13],
    inset_yticks=[40, 50, 60, 70, 80, 90],
    output_file="existential_inclusive.pdf"
)
plot_metric_second(
    filename="existential_inclusive2.tsv",
    ylabel="Inclusive existential accuracy (%)",
    title="Inclusive Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    inset_xlim=(8.5, 13.5),
    inset_ylim=(40, 90),
    inset_xticks=[9, 10, 11, 12, 13],
    inset_yticks=[40, 50, 60, 70, 80, 90],
    output_file="existential_inclusive2.pdf",
    subgraph = False,
)

"""Existential Extrapolated Accuracy"""
plot_metric_full(
    filename="existential_extrapolated.tsv",
    ylabel="Extrapolated existential accuracy (%)",
    title="Extrapolated Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="existential_extrapolated.pdf"
)
plot_metric_full_second(
    filename="existential_extrapolated2.tsv",
    ylabel="Extrapolated existential accuracy (%)",
    title="Extrapolated Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="existential_extrapolated2.pdf"
)

"""Explicit Accuracy"""
plot_metric_full(
    filename="explicit.tsv",
    ylabel="Explicit accuracy (%)",
    title="Explicit Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="explicit.pdf"
)
plot_metric_full_second(
    filename="explicit2.tsv",
    ylabel="Explicit accuracy (%)",
    title="Explicit Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="explicit2.pdf"
)

"""Extrapolated Accuracy"""
plot_metric_full(
    filename="extrapolated.tsv",
    ylabel="Extrapolated accuracy (%)",
    title="Extrapolated Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="extrapolated.pdf"
)
plot_metric_full_second(
    filename="extrapolated2.tsv",
    ylabel="Extrapolated accuracy (%)",
    title="Extrapolated Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="extrapolated2.pdf"
)

"""Inclusive Uniform Accuracy"""
plot_metric_full(
    filename="inclusive_uniform.tsv",
    ylabel="Inclusive Uniform accuracy (%)",
    title="Inclusive Uniform Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_uniform.pdf"
)
plot_metric_full_second(
    filename="inclusive_uniform2.tsv",
    ylabel="Inclusive Uniform accuracy (%)",
    title="Inclusive Uniform Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_uniform2.pdf"
)

"""Inclusive Proportional Accuracy"""
plot_metric_full(
    filename="inclusive_proportional.tsv",
    ylabel="Inclusive Proportional accuracy (%)",
    title="Inclusive Proportional Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_proportional.pdf"
)
plot_metric_full_second(
    filename="inclusive_proportional2.tsv",
    ylabel="Inclusive Proportional accuracy (%)",
    title="Inclusive Proportional Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_proportional2.pdf"
)

"""Rule Count Comparison - Proposed vs Environment"""
plot_rule_counts(
    filename="proposed_env.tsv",
    ylabel="Number of rules",
    title="Proposed vs Environmental Rules Across Independent Test Cycles\nfor Different Configurations",
    output_file="proposed_env.pdf"
)
plot_rule_counts_second(
    filename="proposed_env2.tsv",
    ylabel="Number of rules",
    title="Proposed vs Environmental Rules Across Independent Test Cycles\nfor Different Configurations",
    output_file="proposed_env2.pdf"
)

"""Overall plotting, percentages and counts"""
plot_overall_percentages(
    filename="overall_percentages.tsv",
    ylabel="Mean accuracy / satisfaction (%)",
    title="Overall Mean Percentage Metrics Across Configurations",
    output_file="overall_percentages.pdf"
)

plot_overall_counts(
    filename="overall_counts.tsv",
    ylabel="Mean environmental rules",
    title="Mean Environmental Rules by Configuration",
    output_file="overall_counts.pdf"
)