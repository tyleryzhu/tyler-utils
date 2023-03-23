""" Examples of plots generated from matplotlib"""
from wsgiref.handlers import IISCGIHandler
import numpy as np
from cycler import cycler
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# original: blue, orange, green, red, purple, brown, pink, grey, lime, cyan
# blue: #4169E1
# orange: #F26419
# green: #63B42D
# red: #D23751
# yellow: #F5AD29
# purple: #A4036F
# brown: #76362E
# pink: #E6BCCD or #D295BF
# grey: #5D675B
# cyan: #55DDE0

# Other blue options: dark cobalt, #31348B; #003399
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
tz_color_cycle = cycler(color=tz_colors)


def color_compare(colors=None):
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    lwbase = plt.rcParams["lines.linewidth"]
    thin = lwbase / 2
    thick = lwbase * 3

    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for icol in range(2):
        if icol == 0:
            lwx, lwy = thin, lwbase
        else:
            lwx, lwy = lwbase, thick
        for irow in range(2):
            for i, color in enumerate(colors):
                axs[irow, icol].axhline(i, color=color, lw=lwx)
                axs[irow, icol].axvline(i, color=color, lw=lwy)

        axs[1, icol].set_facecolor("k")
        axs[1, icol].xaxis.set_ticks(np.arange(0, 10, 2))
        axs[0, icol].set_title(
            "line widths (pts): %g, %g" % (lwx, lwy), fontsize="medium"
        )

    for irow in range(2):
        axs[irow, 0].yaxis.set_ticks(np.arange(0, 10, 2))

    fig.suptitle("Colors in the default prop_cycle", fontsize="large")
    plt.show()


def sinusoidal_plot(color_palette=["r", "k", "c"]):
    # Set the default color cycle
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(color=color_palette)

    x = np.linspace(0, 20, 100)

    fig, axes = plt.subplots(nrows=2)

    for i in range(10):
        axes[0].plot(x, i * (x - 10) ** 2)

    for i in range(10):
        axes[1].plot(x, i * np.cos(x))

    plt.show()


def sinusoidal_simple_plot(custom_colors=["c", "m", "y", "k"]):
    x = np.linspace(0, 2 * np.pi, 50)
    offsets = np.linspace(0, 2 * np.pi, 4, endpoint=False)
    yy = np.transpose([np.sin(x + phi) for phi in offsets])

    default_cycler = cycler(color=["r", "g", "b", "y"]) + cycler(
        linestyle=["-", "--", ":", "-."]
    )

    plt.rc("lines", linewidth=4)
    plt.rc("axes", prop_cycle=default_cycler)

    custom_cycler = cycler(color=custom_colors)

    fig, (ax0, ax1) = plt.subplots(nrows=2)
    ax0.plot(yy)
    ax0.set_title("Set default color cycle to rgby")
    ax1.set_prop_cycle(custom_cycler)
    ax1.plot(yy)
    ax1.set_title("Set axes color cycle to cmyk")

    # Add a bit more space between the two plots.
    fig.subplots_adjust(hspace=0.3)
    plt.show()


if __name__ == "__main__":
    # sinusoidal_plot(color_palette=tz_colors_dark)
    # sinusoidal_simple_plot(custom_colors=tz_colors_dark)
    color_compare(colors=tz_colors)
