import contextlib
import dataclasses

import matplotlib as mpl
import plotly.io as pio
from plotly import graph_objects as go

from opentak.tak_theme import palettes
from opentak.tak_theme._colors import (
    BACKGROUND,
    BLACK,
    BLUE,
    BLUE_CRAYOLA,
    GRID,
    LABEL_TICK,
)

BASE_FONT = "Barlow SemiBold, sans-serif"

HEAD_FONT = "Montserrat, sans-serif"

base_template = go.layout.Template(
    layout=go.Layout(
        title={
            "x": 0,
            "font": {
                "family": HEAD_FONT,
                "color": BLACK,
            },
        },
        font={"family": HEAD_FONT},
        xaxis={
            "automargin": True,
            "tickfont": {
                "family": BASE_FONT,
                "color": LABEL_TICK,
            },
            "gridcolor": GRID,
            "gridwidth": 1,
            "color": LABEL_TICK,
            "title": {
                "font": {
                    "family": HEAD_FONT,
                    "color": LABEL_TICK,
                }
            },
        },
        yaxis={
            "automargin": True,
            "tickfont": {
                "family": BASE_FONT,
                "color": LABEL_TICK,
            },
            "gridcolor": GRID,
            "color": LABEL_TICK,
            "title": {
                "font": {
                    "family": HEAD_FONT,
                    "color": LABEL_TICK,
                }
            },
        },
        hoverlabel={"font": {"family": BASE_FONT, "size": 12}},
        bargap=0.2,
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        colorway=palettes.qualitative.default,
        # type ignore because colorscale cannot be a dictionnary according to plotly-stubs,
        # despite the fact that it works perfectly at runtime.
        colorscale={  # type: ignore[arg-type]
            "sequential": palettes.sequential.light_blues,
            "diverging": palettes.diverging.onoff,
        },
        images=[{"name": "base_template"}],
        modebar={
            "bgcolor": BACKGROUND,
            "color": BLUE_CRAYOLA,
            "activecolor": BLUE,
        },
        legend={"title": {"font": {"color": BLACK}}, "font": {"color": BLACK}},
        template={"data": {"heatmap": [{"autocolorscale": True}]}},
    )
)

pio.templates["tak_theme"] = base_template


def convert_palettes_to_mpl():
    for palette_collection in (palettes.sequential, palettes.diverging):
        for name, palette in dataclasses.asdict(palette_collection).items():
            cmap = mpl.colors.LinearSegmentedColormap.from_list(
                name, [p[1] for p in palette]
            )
            mpl.colormaps.register(cmap=cmap)
            rev = cmap.reversed(name=f"{cmap.name}_r")
            mpl.colormaps.register(cmap=rev)


with contextlib.suppress(ImportError):
    convert_palettes_to_mpl()
