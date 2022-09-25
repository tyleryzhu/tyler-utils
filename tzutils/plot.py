from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


def tb_smooth(scalars: List[float], weight: float) -> List[float]:
    """
    Tensorboard style smoothing. Weight between 0 and 1.
    Large values = more smoothing. 0.8, 0.9 are good defaults.
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def quick_plot(
    ax: Axes,
    df: pd.DataFrame,
    x_key: str,
    y_key: str,
    label_key: str = None,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    smooth: bool = False,
    smooth_weight: float = 0.8,
    param_dict: dict = {},
):
    """
    A helper function to make a quick graph. Figure creation and saving is left to the user.

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    df : pd.DataFrame
        The dataframe containing plot data

    param_dict : dict
       Dictionary of keyword arguments to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    # TODO: Add a custom set of colors to use by default (royalblue)
    labels = [None]
    if label_key:
        labels = df[label_key].unique()
    for label in labels:
        if label:
            data = df[df[label_key] == label]
        else:
            data = df

        if smooth:
            artists = ax.plot(
                data[x_key],
                tb_smooth(data[y_key], smooth_weight),
                linestyle="-",
                alpha=0.8,
                label=label,
            )
            c = artists[0].get_color()
            ax.plot(
                data[x_key], data[y_key], linestyle="-", alpha=0.2, label=label, color=c
            )
        else:
            ax.plot(data[x_key], data[y_key], linestyle="-", alpha=0.8, label=label)

    # plt.grid(which="minor", linestyle="--", linewidth=0.5)
    # plt.grid(which="major", linestyle="-")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return


def add_horizontal_line(
    ax: Axes, x, y, text, color="black", linestyle="--", linewidth=1, alpha=0.8
):
    ax.axhline(y=y, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    y_bot, y_top = ax.get_ylim()
    height = y_top - y_bot
    ax.text(x, y + 0.01 * height, text, rotation=360, color=color)


def make_plots(df):
    fig, ax = plt.subplot(1, 1)
    fig(figsize=(8, 8))
    quick_plot(ax, df, "x", "y")
    fig.savefig("assets/plot.png", bbox_inches="tight")


def multilegend_plot_example(df):
    """Make plots with 2 dimensions of legends, one for bs one for lr."""
    plt.figure(figsize=(8, 8))
    bs = df["bs"].unique()
    lr = df["lr"].unique()
    bs_cols = [plt.get_cmap("Set1")(i) for i in np.linspace(0, 0.9, len(bs))]
    line_styles = [":", "--", "-"]
    lines = []

    for b, c in zip(bs, bs_cols):
        for r, ls in zip(lr, line_styles):
            data = df[(df.bs == b) & (df.lr == r)]
            # Tensorboard style smoothing
            (l,) = plt.plot(
                data["iter"],
                tb_smooth(data["eval_return"], 0.9),
                linestyle=ls,
                color=c,
                alpha=0.8,
            )
            plt.plot(
                data["iter"],
                data["eval_return"],
                linestyle=ls,
                color=c,
                alpha=0.2,
            )
            lines.append(l)

    color_leg = plt.legend(
        lines[2 :: len(lr)], [f"bs={b}" for b in bs], loc="upper center"
    )
    ls_leg = [
        Line2D([0], [0], color="k", ls=ls, marker=".", label=f"lr={r}")
        for r, ls in zip(lr, line_styles)
    ]
    plt.legend(handles=ls_leg, loc="upper left")
    plt.gca().add_artist(color_leg)

    plt.grid(which="minor", linestyle="--", linewidth=0.5)
    plt.grid(which="major", linestyle="-")
    plt.title(f"Hyperparameter Search on HalfCheetah")
    plt.xlabel("Iteration")
    plt.ylabel("Average Eval Return")
    plt.savefig(f"assets/exp4_search.png", bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    # example of how to use this
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5],
            "y": [1, 2, 3, 4, 5],
        }
    )
    df["y2"] = df["y"] * 2
    df["y3"] = df["y"] * 3

    fig, ax = plt.subplot(1, 1)
    fig(figsize=(8, 8))
    quick_plot(ax, df, "x", "y")
