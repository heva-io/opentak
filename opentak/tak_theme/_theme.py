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
        colorscale={
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
