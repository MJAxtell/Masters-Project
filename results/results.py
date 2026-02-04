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
    configs = ["D", "E", "F"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    start_idx = 3
    rotated_colors = colors[start_idx:] + colors[:start_idx]

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

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    start_idx = 3
    rotated_colors = colors[start_idx:] + colors[:start_idx]

    for label, color in zip(configs, rotated_colors):
        axins.plot(
            x, existential_inclusive[label],
            marker="o",
            linewidth=1,
            markersize=3,
            color=color,
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
    configs = ["A", "B", "C"]

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
    x = np.arange(1, len(existential_inclusive) + 1)
    configs = ["D", "E", "F"]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    start_idx = 3
    rotated_colors = colors[start_idx:] + colors[:start_idx]

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

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_overall_percentages(
    filename,
    ylabel,
    title,
    output_file
):
    df = pd.read_csv(filename, sep="\t")

    metrics = df["metric"].values
    configs = ["A", "B", "C", "D", "E", "F"]

    num_metrics = len(metrics)
    x = np.arange(num_metrics)

    base_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    colors = (base_colors * ((num_metrics // len(base_colors)) + 1))[:num_metrics]

    legend_handles = [
        Patch(facecolor=colors[i], label=metrics[i])
        for i in range(num_metrics)
    ]

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

    legend_ax = fig.add_subplot(gs[:, 3])
    legend_ax.axis("off")

    legend_ax.legend(
        handles=legend_handles,
        bbox_to_anchor=(1.75, 0.75),
        frameon=True,
        title="Metric"
    )

    fig.suptitle(title, fontsize=13)
    fig.text(0.04, 0.5, ylabel, va="center", rotation="vertical")

    plt.savefig(output_file)
    plt.close(fig)

def plot_overall_counts(
    filename,
    ylabel,
    title,
    output_file
):
    df = pd.read_csv(filename, sep="\t")

    configs = df.columns[1:]

    values_env = df.iloc[0, 1:].astype(float)
    values_prop = df.iloc[1, 1:].astype(float)

    x = np.arange(len(configs))
    width = 0.35

    plt.figure(figsize=(7.5, 4.5))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    bars_env = plt.bar(
        x - width / 2,
        values_env,
        width,
        label="Environmental rules",
        color=colors[0],
        alpha=0.85
    )

    bars_prop = plt.bar(
        x + width / 2,
        values_prop,
        width,
        label="Proposed rules",
        color=colors[1],
        alpha=0.85
    )

    plt.xlabel("Configuration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, configs)
    plt.legend(frameon=True)

    for bars in (bars_env, bars_prop):
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

def fraction_series_to_decimal(series):
    return series.apply(
        lambda v: float(v.split("/")[0].strip()) / float(v.split("/")[1].strip())
    )
def plot_rule_satisfaction_generic(
    filename,
    configs,
    ylabel,
    title,
    output_file,
    rotate_colors=False,
):
    df = pd.read_csv(filename, sep="\t")

    x = np.arange(1, len(df) + 1)

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if rotate_colors:
        colors = colors[3:] + colors[:3]

    plt.figure(figsize=(7.5, 4.5))

    for cfg, color in zip(configs, colors):
        satisfied = fraction_series_to_decimal(df[f"{cfg}-Satisfied"]) * 100
        matched   = fraction_series_to_decimal(df[f"{cfg}-Matched"]) * 100

        plt.plot(
            x,
            satisfied,
            marker="o",
            linewidth=1.4,
            markersize=5,
            alpha=0.8,
            color=color,
            label=f"{cfg} satisfied"
        )

        plt.plot(
            x,
            matched,
            linestyle="--",
            linewidth=1.4,
            alpha=0.8,
            color=color,
            label=f"{cfg} matched"
        )

    plt.xticks(x)
    plt.ylim(0, 105)

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

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

"""Existential Inclusive Accuracy"""
plot_metric_second(
    filename="existential_inclusive.tsv",
    ylabel="Inclusive existential accuracy (%)",
    title="Inclusive Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    inset_xlim=(8.5, 13.5),
    inset_ylim=(40, 90),
    inset_xticks=[9, 10, 11, 12, 13],
    inset_yticks=[40, 50, 60, 70, 80, 90],
    output_file="existential_inclusive.png",
    subgraph=False,
)
plot_metric(
    filename="existential_inclusive2.tsv",
    ylabel="Inclusive existential accuracy (%)",
    title="Inclusive Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    inset_xlim=(8.5, 13.5),
    inset_ylim=(40, 90),
    inset_xticks=[9, 10, 11, 12, 13],
    inset_yticks=[40, 50, 60, 70, 80, 90],
    output_file="existential_inclusive2.png"
)

"""Existential Extrapolated Accuracy"""
plot_metric_full(
    filename="existential_extrapolated.tsv",
    ylabel="Extrapolated existential accuracy (%)",
    title="Extrapolated Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="existential_extrapolated.png"
)
plot_metric_full_second(
    filename="existential_extrapolated2.tsv",
    ylabel="Extrapolated existential accuracy (%)",
    title="Extrapolated Existential Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="existential_extrapolated2.png"
)

"""Explicit Accuracy"""
plot_metric_full(
    filename="explicit.tsv",
    ylabel="Explicit accuracy (%)",
    title="Explicit Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="explicit.png"
)
plot_metric_full_second(
    filename="explicit2.tsv",
    ylabel="Explicit accuracy (%)",
    title="Explicit Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="explicit2.png"
)

"""Extrapolated Accuracy"""
plot_metric_full(
    filename="extrapolated.tsv",
    ylabel="Extrapolated accuracy (%)",
    title="Extrapolated Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="extrapolated.png"
)
plot_metric_full_second(
    filename="extrapolated2.tsv",
    ylabel="Extrapolated accuracy (%)",
    title="Extrapolated Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="extrapolated2.png"
)

"""Inclusive Uniform Accuracy"""
plot_metric_full(
    filename="inclusive_uniform.tsv",
    ylabel="Inclusive Uniform accuracy (%)",
    title="Inclusive Uniform Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_uniform.png"
)
plot_metric_full_second(
    filename="inclusive_uniform2.tsv",
    ylabel="Inclusive Uniform accuracy (%)",
    title="Inclusive Uniform Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_uniform2.png"
)

"""Inclusive Proportional Accuracy"""
plot_metric_full(
    filename="inclusive_proportional.tsv",
    ylabel="Inclusive Proportional accuracy (%)",
    title="Inclusive Proportional Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_proportional.png"
)
plot_metric_full_second(
    filename="inclusive_proportional2.tsv",
    ylabel="Inclusive Proportional accuracy (%)",
    title="Inclusive Proportional Accuracy Across Independent Test Cycles\nfor Different Configurations",
    output_file="inclusive_proportional2.png"
)

"""Rule Count Comparison - Proposed vs Environment"""
plot_rule_counts(
    filename="proposed_env.tsv",
    ylabel="Number of rules",
    title="Proposed vs Environmental Rules Across Independent Test Cycles\nfor Different Configurations",
    output_file="proposed_env.png"
)
plot_rule_counts_second(
    filename="proposed_env2.tsv",
    ylabel="Number of rules",
    title="Proposed vs Environmental Rules Across Independent Test Cycles\nfor Different Configurations",
    output_file="proposed_env2.png"
)

"""Overall plotting, percentages and counts"""
plot_overall_percentages(
    filename="overall_percentages.tsv",
    ylabel="Mean accuracy / satisfaction (%)",
    title="Overall Mean Percentage Metrics Across Configurations",
    output_file="overall_percentages.png"
)

plot_overall_counts(
    filename="overall_counts.tsv",
    ylabel="Mean environmental rules",
    title="Mean Environmental Rules by Configuration",
    output_file="overall_counts.png"
)

plot_rule_satisfaction_generic(
    filename="rule_satisfaction.tsv",
    configs=["A", "B", "C"],
    ylabel="Rule satisfaction / matching (%)",
    title="Rule Satisfaction and Matching Across Test Cycles\nfor Different Configurations",
    output_file="rule_satisfaction.png",
    rotate_colors=False,
)

plot_rule_satisfaction_generic(
    filename="rule_satisfaction2.tsv",
    configs=["D", "E", "F"],
    ylabel="Rule satisfaction / matching (%)",
    title="Rule Satisfaction and Matching Across Test Cycles\nfor Different Configurations",
    output_file="rule_satisfaction2.png",
    rotate_colors=True,
)