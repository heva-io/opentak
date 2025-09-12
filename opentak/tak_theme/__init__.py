"""OPENTAK plotly theme."""

from webcolors import hex_to_rgb

from opentak.tak_theme import palettes
from opentak.tak_theme._theme import base_template
from opentak.tak_theme._utils import apply_mpl_style, lighten_color, set_style

__all__ = (
    "apply_mpl_style",
    "base_template",
    "hex_to_rgb",
    "lighten_color",
    "palettes",
    "set_style",
)
