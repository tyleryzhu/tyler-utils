from re import I
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

# Custom colors
tz_colors = [
    "#4169E1",
    "#F26419",
    "#63B42D",
    "#D23751",
    "#F5AD29",
    "#A4036F",
    "#76362E",
    "#D295BF",
    "#5D675B",
    "#55DDE0",
]
mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=tz_colors)


def tb_smooth(scalars: List[float], weight: float) -> List[float]:
    """
    Tensorboard style smoothing. Weight between 0 and 1.
    Large values = more smoothing. 0.8, 0.9 are good defaults.
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + \
            (1 - weight) * point  # Calculate smoothed value
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
    Helper plotting function. Input dataframe is vertical, separated by a label column, i.e. varied x's and y's.
    Figure creation and saving is left to the user.

    Ex:
      x  |  y  | label 
    -------------------------
      1  |  2  | double 
      2  |  4  | double
      3  |  6  | double
      1  |  3  | triple
      2  |  6  | triple
      4  |  12 | triple

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
            ax.plot(data[x_key], data[y_key],
                    linestyle="-", alpha=0.2, color=c)
        else:
            ax.plot(data[x_key], data[y_key],
                    linestyle="-", alpha=0.8, label=label)

    # plt.grid(which="minor", linestyle="--", linewidth=0.5)
    # plt.grid(which="major", linestyle="-")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return


def quick_plot_y(
    ax: Axes,
    data: pd.DataFrame,
    x_key: str,
    y_keys: str,
    labels: List[str] = None,
    title: str = None,
    x_label: str = None,
    y_label: str = None,
    smooth: bool = False,
    smooth_weight: float = 0.8,
):
    """
    Helper plotting function. Input dataframe is horizontal, i.e. shared x keys but multiple y outputs.
    Figure creation and saving is left to the user.

    Ex:
      x  |  y1  |  y2  |  y3
    -------------------------
      1  |  0.1 |  0.2 |  0.3
      2  |  0.4 |  0.5 |  0.6
      3  |  0.7 |  0.8 |  0.9

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
    if labels is None:
        labels = y_keys
    else:
        assert len(labels) == len(y_keys)
    for y_key, label in zip(y_keys, labels):
        if smooth:
            artists = ax.plot(
                data[x_key],
                tb_smooth(data[y_key], smooth_weight),
                linestyle="-",
                alpha=0.8,
                label=label,
            )
            c = artists[0].get_color()
            ax.plot(data[x_key], data[y_key],
                    linestyle="-", alpha=0.2, color=c)
        else:
            ax.plot(data[x_key], data[y_key],
                    linestyle="-", alpha=0.8, label=label)

    # plt.grid(which="minor", linestyle="--", linewidth=0.5)
    # plt.grid(which="major", linestyle="-")
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    return


def quick_plot_std_error(
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
    std_weight: float = 1.0,
):
    """
    Helper plotting function. Input dataframe is vertical, separated by a label column, i.e. varied x's and y's.
    Includes a std bar based on joining over similar entries which share a join_key (like a seed).
    Figure creation and saving is left to the user.

    Ex:
      x  |  y  | label  | seed
    -------------------------
      1  |   2 | double | 0 
      2  |   4 | double | 0
      1  | 2.5 | double | 1
      2  |   5 | double | 1

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    df : pd.DataFrame
        The dataframe containing plot data

    Returns
    -------
    out : list
        list of artists added
    """
    labels = [None]
    if label_key:
        labels = df[label_key].unique()
    for label in labels:
        if label:
            data = df[df[label_key] == label]
        else:
            data = df

        # Get the mean and std of data that shares join_key
        x = data[x_key].unique()
        mean = data.groupby(x_key)[y_key].mean()
        std = data.groupby(x_key)[y_key].std()
        # Plot mean
        if smooth:
            artists = ax.plot(
                x,
                tb_smooth(mean, smooth_weight),
                linestyle="-",
                alpha=0.8,
                label=label,
            )
            c = artists[0].get_color()
            ax.plot(x, mean, linestyle="-", alpha=0.2, color=c)
        else:
            artists = ax.plot(x, mean,
                              linestyle="-", alpha=0.8, label=label)
            c = artists[0].get_color()

        # Then add std error bars
        ax.fill_between(
            x,
            mean - std_weight * std,
            mean + std_weight * std,
            alpha=0.1,
            color=c,
        )

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
    ax.axhline(y=y, color=color, linestyle=linestyle,
               linewidth=linewidth, alpha=alpha)
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
        lines[2:: len(lr)], [f"bs={b}" for b in bs], loc="upper center"
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
    df = pd.DataFrame({"x": np.linspace(0.01, 2, 100)})
    df["y"] = df["x"]
    df["y2"] = df["x"] ** 2
    df["y3"] = df["x"] ** 3 + np.random.normal(0, 0.6, 100)
    df["y_sqrt"] = np.sqrt(df["x"])
    df["y_qrt"] = df["x"] ** 0.25
    df["y_log"] = np.log(df["x"])
    df["y_sin"] = np.sin(df["x"] * np.pi) * 4
    df["y_cos"] = np.cos(df["x"] * np.pi) * 4
    df["y_scs"] = np.sin((df["x"] - 0.5) * np.pi) * 4
    df["y_scc"] = np.sin((df["x"] - 1) * np.pi) * 4
    y_keys = [
        "y",
        "y2",
#        "y3",
        "y_sqrt",
        "y_qrt",
        "y_log",
        "y_sin",
        "y_cos",
        "y_scs",
        "y_scc",
    ]
    labels = [
        "linear",
        "quadratic",
#        "cubic",
        "sqrt",
        "qrt",
        "log",
        "sin",
        "cos",
        "sin2",
        "sin3",
    ]

    plt.figure(figsize=(8, 8))
    ax = plt.subplot()
    quick_plot_y(ax, df, "x", y_keys, labels, "Example Plot")
    quick_plot_y(ax, df, "x", ["y3"], ["noisy_cubic"], "Example Plot", smooth=True)
    plt.savefig("example.png", bbox_inches="tight")
    plt.show()
