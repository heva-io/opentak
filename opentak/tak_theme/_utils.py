from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import plotly.io as pio
import webcolors
from coloraide import Color

if TYPE_CHECKING:
    from collections.abc import Sequence

    from matplotlib.figure import Figure

import contextlib

import matplotlib.pyplot as plt

from opentak.tak_theme._colors import LABEL_TICK


def lighten_color(color: Sequence | str, opacity: float = 0.5) -> str:
    """Change the opacity of a color.

    :param color: trace color
    :param opacity: opacity factor
    :return: less opaque color as a string for plotly usage
    """
    if isinstance(color, str) and color.startswith("#"):
        fill_color_tuple = webcolors.hex_to_rgb(color)
    elif isinstance(color, str):
        fill_color_tuple = webcolors.name_to_rgb(color)
    else:
        fill_color_tuple = color

    return f"rgba{(*fill_color_tuple, opacity)}"


def interpolate_colors(
    color_list: list[str], nb_colors: int, return_steps: bool = False
) -> list[str] | tuple[list[str], list[float]]:
    """Given a list of colors, create a discrete gradient.

    :param color_list: list of colors
    :param nb_colors: nb of colors to get
    :param return_steps: return float steps [0, 1].

    :return: list of colors containing plus new colors interpolated
    """
    # similar to chroma web tool
    space_interpolation = "srgb"

    if len(color_list) < 2:
        raise ValueError("color_list should contain at least two colors")

    if nb_colors < len(color_list):
        raise ValueError("nb_colors should be higher than color_list")

    interpolator = Color.interpolate(color_list, space=space_interpolation)
    steps = cast("list[float]", np.linspace(0, 1, nb_colors).tolist())
    interpolated = [interpolator(x).to_string(hex=True) for x in steps]

    if return_steps:
        return interpolated, steps
    return interpolated


def apply_mpl_style(fig: Figure | None = None) -> None:
    """Apply theme to a matplotlib figure."""
    if fig is None:
        fig = plt.gcf()

    axs = fig.axes
    for ax in axs:
        ax.tick_params(axis="both", colors=LABEL_TICK)
        ax.xaxis.label.set_color(LABEL_TICK)
        ax.yaxis.label.set_color(LABEL_TICK)
        ax.grid(visible=True)
        # type ignore because the type hints for matplotlib are not up to date
        for tick in ax.get_xticklabels():  # type: ignore[operator]
            tick.set_fontname("Barlow")
            tick.set_fontweight("semibold")
        for tick in ax.get_yticklabels():  # type: ignore[operator]
            tick.set_fontname("Barlow")
            tick.set_fontweight("semibold")
    fig.tight_layout()


def set_style() -> None:
    """Set plotly default style to 'tak_theme'."""
    pio.templates.default = "tak_theme"
    with contextlib.suppress(ImportError):
        plt.style.use("opentak.tak_theme.mpl_tak")  # type: ignore[attr-defined]
