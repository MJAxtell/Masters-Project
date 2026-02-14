"""AI Use-Disclaimer. Generative AI, specifically, ChatGPT, has been used to help with some syntax
in quickly constructing MatplotLib plots. All code is handwritten and reviewed and all data manually collected
from the training processes."""
import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.gridspec import GridSpec

BLUES = {
    "A": "#0B1C2D",
    "B": "#1F4ED8",
    "C": "#00B4D8",
}

REDS = {
    "D": "#5A0505",
    "E": "#D62828",
    "F": "#6A5ACD",
}

EXPANDED_BLUES = [
    "#1F2A44",
    "#243A5E",
    "#2A4D69",
    "#356F8C",
    "#4A8FB5",
    "#6FAED6",
    "#9BC7E4",
    "#C7DEF0",
]

EXPANDED_REDS = [
    "#5B0F14",
    "#7A1418",
    "#9E1B1F",
    "#C62828",
    "#E53935",
    "#EF5350",
]

EXPANDED_PURPLES = [
    "#2E0B24",
    "#4A1238",
    "#6B1F52",
    "#8F3A74",
    "#B45A96",
    "#D88CBF",
]

def plot_metric(
    filename,
    configs,
    colours,
    ylabel,
    title,
    output_file,
    *,
    y_limits=None,
    x_ticks=None,
    short=True,
    show_outliers=True,
    outlier_xlim=None,
    outlier_ylim=None,
):
    if len(configs) != len(colours):
        raise ValueError("configs and colours must be the same length")

    df = pd.read_csv(filename, sep="\t")
    x = df["Test Cycle"]

    plt.figure(figsize=(7.5, 4.5))

    for cfg, color in zip(configs, colours):
        plt.plot(
            x,
            df[cfg],
            marker="o",
            linewidth=1.4,
            markersize=5,
            alpha=0.75,
            color=color,
            label=cfg,
        )

    if y_limits is not None:
        y_min, y_max = y_limits

        # visual headroom (no tick at the top)
        y_top = 101 if short else 105
        plt.ylim(y_min, y_top)

        # ticks stop at the true max
        yticks = list(range(int(y_min), int(y_max) + 1, 5))
        if y_max not in yticks:
            yticks.append(y_max)

        plt.yticks(yticks)

    if x_ticks is not None:
        plt.xticks(x_ticks)
    else:
        plt.xticks(range(1, 21))

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.legend(
        title="Configuration",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=True,
    )

    ax = plt.gca()
    ax.grid(True, alpha=0.4)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    if show_outliers:
        axins = inset_axes(
            ax,
            width="35%",
            height="35%",
            loc="lower left",
            bbox_to_anchor=(1.08, 0.1, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
        )

        for cfg, color in zip(configs, colours):
            axins.plot(
                x,
                df[cfg],
                marker="o",
                linewidth=1,
                markersize=3,
                alpha=0.75,
                color=color,
            )

        axins.set_autoscale_on(False)
        axins.set_xlim(*outlier_xlim)
        axins.set_ylim(*outlier_ylim)

        axins.set_title("Outliers", fontsize=8)
        axins.tick_params(axis="both", which="major", labelsize=8)

        axins.grid(True, alpha=0.4)
        for spine in axins.spines.values():
            spine.set_linewidth(0.8)

    plt.subplots_adjust(right=0.73, top=0.88)
    plt.savefig(output_file)
    plt.close()

def plot_rule_counts_blue(
    filename,
    ylabel,
    title,
    output_file,
):
    df = pd.read_csv(filename, sep="\t")
    x = df["Test Cycle"]

    configs = ["A", "B", "C"]

    color_pairs = [
        (BLUES["A"], BLUES["A"]),
        (BLUES["B"], BLUES["B"]),
        (BLUES["C"], BLUES["C"]),
    ]

    plt.figure(figsize=(7.5, 4.5))

    for cfg, (prop_color, env_color) in zip(configs, color_pairs):
        proposed = df[f"{cfg}_proposed"]
        env      = df[f"{cfg}_env"]

        plt.plot(
            x,
            proposed,
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=prop_color,
            label=f"{cfg} proposed",
        )

        plt.plot(
            x,
            env,
            linestyle="-.",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=env_color,
            label=f"{cfg} environment",
        )
    plt.xticks(range(1, 21))
    plt.xlim(0.5, 20.5)

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_rule_counts_red(
    filename,
    ylabel,
    title,
    output_file,
):
    df = pd.read_csv(filename, sep="\t")
    x = df["Test Cycle"]

    configs = ["D", "E", "F"]

    color_pairs = [
        (REDS["D"], REDS["D"]),
        (REDS["E"], REDS["E"]),
        (REDS["F"], REDS["F"]),
    ]

    plt.figure(figsize=(7.5, 4.5))

    for cfg, (prop_color, env_color) in zip(configs, color_pairs):
        proposed = df[f"{cfg}_proposed"]
        env      = df[f"{cfg}_env"]

        plt.plot(
            x,
            proposed,
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=prop_color,
            label=f"{cfg} proposed",
        )

        plt.plot(
            x,
            env,
            linestyle="-.",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=env_color,
            label=f"{cfg} environment",
        )
    plt.xticks(range(1, 21))
    plt.xlim(0.5, 20.5)

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
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

    colors = EXPANDED_BLUES

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

    values_env  = df.iloc[0, 1:].astype(float)
    values_prop = df.iloc[1, 1:].astype(float)

    x = np.arange(len(configs))
    width = 0.35

    plt.figure(figsize=(7.5, 4.5))

    # Cycle colours safely
    blues = itertools.cycle(EXPANDED_REDS)
    reds  = itertools.cycle(EXPANDED_PURPLES)

    bars_env = []
    bars_prop = []

    for i, cfg in enumerate(configs):
        color_env = next(blues)
        color_prop = next(reds)

        bars_env.append(
            plt.bar(
                x[i] - width / 2,
                values_env.iloc[i],
                width,
                color=color_env,
                alpha=0.85,
                label="Environmental rules" if i == 0 else None
            )[0]
        )

        bars_prop.append(
            plt.bar(
                x[i] + width / 2,
                values_prop.iloc[i],
                width,
                color=color_prop,
                alpha=0.85,
                label="Proposed rules" if i == 0 else None
            )[0]
        )

    plt.xlabel("Configuration")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(x, configs)
    plt.legend(frameon=True)

    # ---- Value labels ----
    for bar in bars_env + bars_prop:
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
def plot_rule_satisfaction_blue(
    filename,
    configs,
    ylabel,
    title,
    output_file,
):
    df = pd.read_csv(filename, sep="\t")
    x = np.arange(1, len(df) + 1)

    color_pairs = [
        (BLUES["A"], BLUES["A"]),
        (BLUES["B"], BLUES["B"]),
        (BLUES["C"], BLUES["C"]),
    ]

    plt.figure(figsize=(7.5, 4.5))

    for cfg, (sat_color, match_color) in zip(configs, itertools.cycle(color_pairs)):
        satisfied = fraction_series_to_decimal(df[f"{cfg}-Satisfied"]) * 100
        matched   = fraction_series_to_decimal(df[f"{cfg}-Matched"]) * 100

        plt.plot(
            x,
            satisfied,
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=sat_color,
            label=f"{cfg} satisfied",
        )

        plt.plot(
            x,
            matched,
            linestyle="-.",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=match_color,
            label=f"{cfg} matched",
        )
    plt.xticks(range(1, 21))
    plt.xlim(0.5, 20.5)

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_rule_satisfaction_red(
    filename,
    configs,
    ylabel,
    title,
    output_file,
):
    df = pd.read_csv(filename, sep="\t")
    x = np.arange(1, len(df) + 1)

    color_pairs = [
        (REDS["D"], REDS["D"]),
        (REDS["E"], REDS["E"]),
        (REDS["F"], REDS["F"]),
    ]

    plt.figure(figsize=(7.5, 4.5))

    for cfg, (sat_color, match_color) in zip(configs, itertools.cycle(color_pairs)):
        satisfied = fraction_series_to_decimal(df[f"{cfg}-Satisfied"]) * 100
        matched   = fraction_series_to_decimal(df[f"{cfg}-Matched"]) * 100

        plt.plot(
            x,
            satisfied,
            linestyle="-",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=sat_color,
            label=f"{cfg} satisfied",
        )

        plt.plot(
            x,
            matched,
            linestyle="-.",
            marker="o",
            markersize=4,
            linewidth=2,
            alpha=0.8,
            color=match_color,
            label=f"{cfg} matched",
        )
    plt.xticks(range(1, 21))
    plt.xlim(0.5, 20.5)

    plt.xlabel("Test cycle (independent runs)")
    plt.ylabel(ylabel)
    plt.title(title)

    plt.grid(
        True,
        which="both",
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

def plot_heatmap_ax(
    ax,
    df,
    title=None,
    show_axes=False,
    z_clip=3.0,
    metric_columns=None,
):
    if metric_columns is None:
        metric_columns = [
            "Inclusive Existential",
            "Extrap Existential",
            "Explicit Acc",
            "Extrap Acc",
            "Inclusive Uniform Acc",
            "Inclusive Proportional Acc",
        ]

    data = df[metric_columns].to_numpy(dtype=float)
    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=0)
    z = np.clip((data - mean) / std, -z_clip, z_clip)

    im = ax.imshow(
        z,
        aspect="auto",
        cmap="coolwarm",
        vmin=-z_clip,
        vmax=z_clip,
    )

    if show_axes:
        ax.set_xticks(np.arange(len(metric_columns)))
        ax.set_xticklabels(metric_columns, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(df)))
        ax.set_yticklabels(df["Test Cycle"])
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    if title:
        ax.set_title(title, fontsize=11)

    return im

def plot_heatmap_grid(
    files,
    titles,
    output_file,
    ncols=3,
    z_clip=3.0,
):
    n = len(files)
    nrows = int(np.ceil(n / ncols))

    fig = plt.figure(figsize=(3.4 * ncols + 1.2, 3.6 * nrows))
    gs = fig.add_gridspec(
        nrows,
        ncols + 1,
        width_ratios=[1] * ncols + [0.08],
        wspace=0.08,
        hspace=0.22,
    )

    axes = []
    ims = []

    for i in range(n):
        r = i // ncols
        c = i % ncols
        ax = fig.add_subplot(gs[r, c])
        axes.append(ax)

        df = pd.read_csv(files[i], sep="\t")

        im = plot_heatmap_ax(
            ax=ax,
            df=df,
            title=titles[i],
            show_axes=(i == (nrows - 1) * ncols),
            z_clip=z_clip,
        )

        ims.append(im)

    cax = fig.add_subplot(gs[:, -1])
    cbar = fig.colorbar(ims[0], cax=cax)
    cbar.set_label("Relative deviation (z-score)", labelpad=14)

    fig.suptitle(
        "Relative Metric Deviations Across Test Cycles",
        fontsize=14,
        y=0.97,
    )

    plt.subplots_adjust(
        left=0.10,
        right=0.92,
        top=0.90,
        bottom=0.18,
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

"""Existential Inclusive Accuracy"""
plot_metric(
    filename="existential_inclusive.tsv",
    configs=["A", "B", "C"],
    colours=[BLUES["A"], BLUES["B"], BLUES["C"]],
    ylabel="Inclusive existential accuracy (%)",
    title="Inclusive Existential Accuracy Across Independent Test Cycles\nfor Configurations A, B, C",
    y_limits=(80, 100),
    short=True,
    show_outliers=False,
    output_file="Graphs/existential_inclusive.png",
)

plot_metric(
    filename="existential_inclusive2.tsv",
    configs=["D", "E", "F"],
    colours=[REDS["D"], REDS["E"], REDS["F"]],
    ylabel="Inclusive existential accuracy (%)",
    title="Inclusive Existential Accuracy Across Independent Test Cycles\nfor Configurations D, E, F",
    y_limits=(80, 100),
    short=True,
    outlier_xlim=(8.5, 13.5),
    outlier_ylim=(40, 90),
    output_file="Graphs/existential_inclusive2.png",
)

"""Existential Extrapolated Accuracy"""
plot_metric(
    filename="existential_extrapolated.tsv",
    configs=["A", "B", "C"],
    colours=[BLUES["A"], BLUES["B"], BLUES["C"]],
    ylabel="Extrapolated existential accuracy (%)",
    title="Extrapolated Existential Accuracy Across Independent Test Cycles\nfor Configurations A, B, C",
    y_limits=(0, 100),
    short=False,
    show_outliers=False,
    output_file="Graphs/existential_extrapolated.png",
)

plot_metric(
    filename="existential_extrapolated2.tsv",
    configs=["D", "E", "F"],
    colours=[REDS["D"], REDS["E"], REDS["F"]],
    ylabel="Extrapolated existential accuracy (%)",
    title="Extrapolated Existential Accuracy Across Independent Test Cycles\nfor Configurations D, E, F",
    y_limits=(0, 100),
    short=False,
    show_outliers=False,
    output_file="Graphs/existential_extrapolated2.png",
)

"""Explicit Accuracy"""
plot_metric(
    filename="explicit.tsv",
    configs=["A", "B", "C"],
    colours=[BLUES["A"], BLUES["B"], BLUES["C"]],
    ylabel="Explicit accuracy (%)",
    title="Explicit Accuracy Across Independent Test Cycles\nfor Configurations A, B, C",
    y_limits=(0, 100),
    short=False,
    show_outliers=False,
    output_file="Graphs/explicit.png",
)

plot_metric(
    filename="explicit2.tsv",
    configs=["D", "E", "F"],
    colours=[REDS["D"], REDS["E"], REDS["F"]],
    ylabel="Explicit accuracy (%)",
    title="Explicit Accuracy Across Independent Test Cycles\nfor Configurations D, E, F",
    y_limits=(0, 100),
    short=False,
    show_outliers=False,
    output_file="Graphs/explicit2.png",
)

"""Extrapolated Accuracy"""
plot_metric(
    filename="extrapolated.tsv",
    configs=["A", "B", "C"],
    colours=[BLUES["A"], BLUES["B"], BLUES["C"]],
    ylabel="Extrapolated accuracy (%)",
    title="Extrapolated Accuracy Across Independent Test Cycles\nfor Configurations A, B, C",
    y_limits=(0, 100),
    short=False,
    show_outliers=False,
    output_file="Graphs/extrapolated.png",
)

plot_metric(
    filename="extrapolated2.tsv",
    configs=["D", "E", "F"],
    colours=[REDS["D"], REDS["E"], REDS["F"]],
    ylabel="Extrapolated accuracy (%)",
    title="Extrapolated Accuracy Across Independent Test Cycles\nfor Configurations D, E, F",
    y_limits=(0, 100),
    short=False,
    show_outliers=False,
    output_file="Graphs/extrapolated2.png",
)

"""Inclusive Uniform Accuracy"""
plot_metric(
    filename="inclusive_uniform.tsv",
    configs=["A", "B", "C"],
    colours=[BLUES["A"], BLUES["B"], BLUES["C"]],
    ylabel="Inclusive Uniform accuracy (%)",
    title="Inclusive Uniform Accuracy Across Independent Test Cycles\nfor Configurations A, B, C",
    y_limits=(80, 100),
    short=True,
    outlier_xlim=(10, 19),
    outlier_ylim=(25, 85),
    output_file="Graphs/inclusive_uniform.png",
)

plot_metric(
    filename="inclusive_uniform2.tsv",
    configs=["D", "E", "F"],
    colours=[REDS["D"], REDS["E"], REDS["F"]],
    ylabel="Inclusive Uniform accuracy (%)",
    title="Inclusive Uniform Accuracy Across Independent Test Cycles\nfor Configurations D, E, F",
    y_limits=(80, 100),
    short=True,
    outlier_xlim=(5, 13),
    outlier_ylim=(60, 85),
    output_file="Graphs/inclusive_uniform2.png",
)

"""Inclusive Proportional Accuracy"""
"""Inclusive Proportional Accuracy — A, B, C"""
plot_metric(
    filename="inclusive_proportional.tsv",
    configs=["A", "B", "C"],
    colours=[BLUES["A"], BLUES["B"], BLUES["C"]],
    ylabel="Inclusive Proportional accuracy (%)",
    title="Inclusive Proportional Accuracy Across Independent Test Cycles\nfor Configurations A, B, C",
    y_limits=(80, 100),
    short=True,
    outlier_xlim=(10, 19),
    outlier_ylim=(20, 85),
    output_file="Graphs/inclusive_proportional.png",
)

plot_metric(
    filename="inclusive_proportional2.tsv",
    configs=["D", "E", "F"],
    colours=[REDS["D"], REDS["E"], REDS["F"]],
    ylabel="Inclusive Proportional accuracy (%)",
    title="Inclusive Proportional Accuracy Across Independent Test Cycles\nfor Configurations D, E, F",
    y_limits=(80, 100),
    short=True,
    outlier_xlim=(0, 13),
    outlier_ylim=(60, 85),
    output_file="Graphs/inclusive_proportional2.png",
)

"""Rule Count Comparison - Proposed vs Environment"""
plot_rule_counts_blue(
    filename="proposed_env.tsv",
    ylabel="Number of rules",
    title="Proposed vs Environmental Rules Across Independent Test Cycles\nfor Configurations A, B, C",
    output_file="Graphs/proposed_env.png"
)
plot_rule_counts_red(
    filename="proposed_env2.tsv",
    ylabel="Number of rules",
    title="Proposed vs Environmental Rules Across Independent Test Cycles\nfor Configurations D, E, F",
    output_file="Graphs/proposed_env2.png"
)

"""Overall plotting, percentages and counts"""
plot_overall_percentages(
    filename="overall_percentages.tsv",
    ylabel="Mean accuracy / satisfaction (%)",
    title="Overall Mean Percentage Metrics Across Configurations",
    output_file="Graphs/overall_percentages.png"
)

plot_overall_counts(
    filename="overall_counts.tsv",
    ylabel="Mean environmental / proposed rules",
    title="Mean Environmental and Proposed Rules by Configuration Across Test Cycles",
    output_file="Graphs/overall_counts.png"
)

plot_rule_satisfaction_blue(
    filename="rule_satisfaction.tsv",
    configs=["A", "B", "C"],
    ylabel="Rule satisfaction / matching (%)",
    title="Environmental Rule Satisfactions and Proposed Rule Matches Across Test Cycles\nfor Configurations A, B, C",
    output_file="Graphs/rule_satisfaction.png",
)

plot_rule_satisfaction_red(
    filename="rule_satisfaction2.tsv",
    configs=["D", "E", "F"],
    ylabel="Rule satisfaction / matching (%)",
    title="Environmental Rule Satisfactions and Proposed Rule Matches Across Test Cycles\nfor Configurations D, E, F",
    output_file="Graphs/rule_satisfaction2.png",
)

"""Heatmaps"""
files = [
    "heatmap1.tsv",
    "heatmap2.tsv",
    "heatmap3.tsv",
    "heatmap4.tsv",
    "heatmap5.tsv",
    "heatmap6.tsv",
]

titles = [
    "Configuration A",
    "Configuration B",
    "Configuration C",
    "Configuration D",
    "Configuration E",
    "Configuration F",
]

plot_heatmap_grid(
    files=files,
    titles=titles,
    output_file="Graphs/heatmap_grid.png",
    z_clip=3.0,
)