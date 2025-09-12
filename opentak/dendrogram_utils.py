from collections import OrderedDict

import numpy as np
import scipy.cluster.hierarchy as sch
from plotly import graph_objs

########################
# Source of this code:poetry
# https://github.com/plotly/plotly.py/blob/master/packages/python/plotly/plotly/figure_factory/_dendrogram.py
# Edited to prevent re-computation of linkage matrix


def create_dendrogram(
    # ruff: noqa: N803
    x: np.ndarray,
    add_sep: bool,
    nb_clusters: int,
    orientation: str = "bottom",
    labels: list[str] | None = None,
    colorscale: list | None = None,
    hovertext: list | None = None,
) -> tuple[graph_objs.Figure, list[float], float]:
    # ruff: noqa: D205
    """Return a dendrogram Plotly figure object. This is a thin
    wrapper around scipy.cluster.hierarchy.dendrogram.

    :param x: Result of scipy.cluster.hierarchy.linkage() function
    :param add_sep: Whether a separation is shown on the TAK or no. This parameter allows the dendogram to be cut and
    only displays clusters on the dendogram
    :param nb_clusters: Number of clusters on the TAK
    :param orientation: 'top', 'right', 'bottom', or 'left'
    :param labels: List of axis category labels(observation labels)
    :param colorscale: Optional colorscale for the dendrogram tree.
                              Requires 8 colors to be specified, the 7th of
                              which is ignored.  With scipy>=1.5.0, the 2nd, 3rd
                              and 6th are used twice as often as the others.
                              Given a shorter list, the missing values are
                              replaced with defaults and with a longer list the
                              extra values are ignored.
    :param hovertext: List of hovertext for constituent traces of dendrogram
                               clusters

    :return: Dendogram representing the hiearchical clustering
    """
    dendrogram = _Dendrogram(
        x,
        add_sep,
        nb_clusters,
        orientation,
        labels,
        colorscale,
        hovertext=hovertext,
    )

    fig = graph_objs.Figure(data=dendrogram.data, layout=dendrogram.layout)

    return (
        fig,
        dendrogram.yaxis_range,
        dendrogram.threshold_vertical,
    )


class _Dendrogram:
    """Refer to create_dendrogram() for docstring."""

    def __init__(
        self,
        # ruff: noqa: N803
        X,
        add_sep: bool,
        nb_clusters: int,
        orientation="bottom",
        labels=None,
        colorscale=None,
        width=np.inf,
        height=np.inf,
        xaxis="xaxis",
        yaxis="yaxis",
        hovertext=None,
    ):
        """Initialize the _Dendrogram class.

        :param x: Result of scipy.cluster.hierarchy.linkage() function
        :param add_sep: Whether a separation is shown on the TAK or not
        :param nb_clusters: Number of clusters on the TAK
        :param orientation: 'top', 'right', 'bottom', or 'left'
        :param labels: List of axis category labels (observation labels)
        :param colorscale: Optional colorscale for the dendrogram tree
        :param width: Width of the figure
        :param height: Height of the figure
        :param xaxis: Name of the x-axis
        :param yaxis: Name of the y-axis
        :param hovertext: List of hovertext for constituent traces of dendrogram clusters
        """
        self.orientation = orientation
        self.add_sep = add_sep
        self.nb_clusters = nb_clusters
        self.labels = labels
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.data = []
        self.leaves = []
        self.sign = {self.xaxis: 1, self.yaxis: 1}
        self.layout: dict[str, dict] = {self.xaxis: {}, self.yaxis: {}}

        if self.orientation in ["left", "bottom"]:
            self.sign[self.xaxis] = 1
        else:
            self.sign[self.xaxis] = -1

        if self.orientation in ["right", "bottom"]:
            self.sign[self.yaxis] = 1
        else:
            self.sign[self.yaxis] = -1

        (
            dd_traces,
            xvals,
            yvals,
            ordered_labels,
            leaves,
            threshold_vertical,
        ) = self.get_dendrogram_traces(X, colorscale, hovertext, self.nb_clusters)

        self.threshold_vertical = threshold_vertical

        self.labels = ordered_labels
        self.leaves = leaves
        yvals_flat = yvals.flatten()
        xvals_flat = xvals.flatten()

        # storing yaxis_range to crop borders of dendrogram
        self.yaxis_range = [0, max(xvals_flat)]

        self.zero_vals: list[float] = []

        for i in range(len(yvals_flat)):
            if yvals_flat[i] == 0 and xvals_flat[i] not in self.zero_vals:
                self.zero_vals.append(xvals_flat[i])

        if len(self.zero_vals) > len(yvals) + 1:
            # If the length of zero_vals is larger than the length of yvals,
            # it means that there are wrong vals because of the identicial samples.
            # Three and more identicial samples will make the yvals of spliting
            # center into 0 and it will accidentally take it as leaves.
            l_border = int(min(self.zero_vals))
            r_border = int(max(self.zero_vals))
            correct_leaves_pos = range(
                l_border, r_border + 1, int((r_border - l_border) / len(yvals))
            )
            # Regenerating the leaves pos from the self.zero_vals with equally intervals.
            self.zero_vals = list(correct_leaves_pos)

        self.zero_vals.sort()
        self.layout = self.set_figure_layout(width, height)
        self.data = dd_traces

    def get_color_dict(self, colorscale):
        """Return colorscale used for dendrogram tree clusters.

        :param colorscale: Colors to use for the plot in rgb format.
        :return: A dict of default colors mapped to the user colorscale.
        """
        # These are the color codes returned for dendrograms
        # We're replacing them with nicer colors
        # This list is the colors that can be used by dendrogram, which were
        # determined as the combination of the default above_threshold_color and
        # the default color palette (see scipy/cluster/hierarchy.py)
        d = {
            "r": "red",
            "g": "green",
            "b": "blue",
            "c": "cyan",
            "m": "magenta",
            "y": "yellow",
            "k": "black",
            # palette in scipy/cluster/hierarchy.py
            "w": "white",
        }
        default_colors = OrderedDict(sorted(d.items(), key=lambda t: t[0]))

        if colorscale is None:
            rgb_colorscale = [
                "rgb(0,0,255)",  # Bleu Docaposte
                "rgb(65,124,255)",  # Bleu secondaire
                "rgb(45,223,213)",  # Bleu turquoise
                "rgb(255,86,87)",  # Corail
                "rgb(255,203,5)",  # Jaune La Poste
                "rgb(112,111,111)",  # Gris La Poste
                "rgb(0,0,0)",  # White
            ]
        else:
            rgb_colorscale = colorscale

        for i in range(len(default_colors.keys())):
            k = list(default_colors.keys())[i]  # PY3 won't index keys
            if i < len(rgb_colorscale):
                default_colors[k] = rgb_colorscale[i]

        # add support for cyclic format colors as introduced in scipy===1.5.0
        # before this, the colors were named 'r', 'b', 'y' etc., now they are
        # named 'C0', 'C1', etc. To keep the colors consistent regardless of the
        # scipy version, we try as much as possible to map the new colors to the
        # old colors
        # this mapping was found by inpecting scipy/cluster/hierarchy.py (see
        # comment above).
        new_old_color_map = [
            ("C0", "b"),
            ("C1", "g"),
            ("C2", "r"),
            ("C3", "c"),
            ("C4", "m"),
            ("C5", "y"),
            ("C6", "k"),
            ("C7", "g"),
            ("C8", "r"),
            ("C9", "c"),
        ]
        for nc, oc in new_old_color_map:
            try:
                default_colors[nc] = default_colors[oc]
            except KeyError:
                # it could happen that the old color isn't found (if a custom
                # colorscale was specified), in this case we set it to an
                # arbitrary default.
                default_colors[nc] = "rgb(0,116,217)"

        return default_colors

    def set_axis_layout(self, axis_key):
        """Set and returns default axis object for dendrogram figure.

        :param (str) axis_key: E.g., 'xaxis', 'xaxis1', 'yaxis', yaxis1', etc.
        :rtype (dict): An axis_key dictionary with set parameters.
        """
        axis_defaults = {
            "type": "linear",
            "ticks": "outside",
            "mirror": "allticks",
            "rangemode": "tozero",
            "showticklabels": True,
            "zeroline": False,
            "showgrid": False,
            "showline": True,
        }

        if len(self.labels) != 0:
            axis_key_labels = self.xaxis
            if self.orientation in ["left", "right"]:
                axis_key_labels = self.yaxis
            if axis_key_labels not in self.layout:
                self.layout[axis_key_labels] = {}
            self.layout[axis_key_labels]["tickvals"] = [
                zv * self.sign[axis_key] for zv in self.zero_vals
            ]
            self.layout[axis_key_labels]["ticktext"] = self.labels
            self.layout[axis_key_labels]["tickmode"] = "array"

        self.layout[axis_key].update(axis_defaults)

        return self.layout[axis_key]

    def set_figure_layout(self, width, height):
        """Set and returns default layout object for dendrogram figure."""
        self.layout.update(
            {
                "showlegend": False,
                "autosize": False,
                "hovermode": "closest",
                "width": width,
                "height": height,
            }
        )

        self.set_axis_layout(self.xaxis)
        self.set_axis_layout(self.yaxis)

        return self.layout

    def get_dendrogram_traces(
        # ruff: noqa: N803
        self,
        x: np.ndarray,
        colorscale: list,
        hovertext: list,
        color_threshold: int,
    ):
        """Calculate all the elements needed for plotting a dendrogram.

        :param x: Linkage matrix
        :param colorscale: Color scale for dendrogram tree clusters
        :param hovertext: List of hovertext for constituent traces of dendrogram
        :param color_threshold: Number of clusters to colour
        :return: All the traces in the following order:
            (a) trace_list: List of Plotly trace objects for dendrogram tree
            (b) icoord: All X points of the dendrogram tree as array of arrays
                with length 4
            (c) dcoord: All Y points of the dendrogram tree as array of arrays
                with length 4
            (d) ordered_labels: leaf labels in the order they are going to
                appear on the plot
            (e) P['leaves']: left-to-right traversal of the leaves
            (f) threshold_vertical: The x coordinate of the threshold vertical line.
        """
        # define an offset so the vline does not overlap nodes of the tree
        offset_vline = 0.1

        dendrogram_dict = sch.dendrogram(
            x,
            color_threshold=x[-(color_threshold - 1), 2],
            orientation=self.orientation,
            labels=self.labels,
            no_plot=True,
        )

        icoord = np.array(dendrogram_dict["icoord"])
        dcoord = np.array(dendrogram_dict["dcoord"])

        ordered_labels = np.array(dendrogram_dict["ivl"])
        color_list = np.array(dendrogram_dict["color_list"])

        colors = self.get_color_dict(colorscale)

        trace_list = []

        # store every x value so after the loop we can rank them
        values_x = set()

        for i in range(len(icoord)):
            # xs and ys are arrays of 4 points that make up the '∩' shapes
            # of the dendrogram tree

            xs = dcoord[i]
            ys = icoord[i]

            color_key = color_list[i]
            hovertext_label = None

            if hovertext:
                hovertext_label = hovertext[i]

            values_x.update(set(xs))

            trace = {
                "type": "scatter",
                "x": np.multiply(self.sign[self.xaxis], xs),
                "y": np.multiply(self.sign[self.yaxis], ys),
                "mode": "lines",
                "marker": {"color": colors[color_key]},
                "text": hovertext_label,
                "hoverinfo": "text",
            }

            try:
                x_index = str(int(self.xaxis[-1]))
            except ValueError:
                x_index = ""

            try:
                y_index = str(int(self.yaxis[-1]))
            except ValueError:
                y_index = ""

            trace["xaxis"] = "x" + x_index
            trace["yaxis"] = "y" + y_index

            trace_list.append(trace)

        # find the threshold value
        # after this value, the subtrees contain the clusters

        if color_threshold == 1:
            threshold_vertical = sorted(values_x)[-color_threshold] + offset_vline
        else:
            last_v = sorted(values_x)[-color_threshold]
            last_v_minus_one = sorted(values_x)[-color_threshold + 1]
            threshold_vertical = last_v - (last_v - last_v_minus_one) / 2

        return (
            trace_list,
            icoord,
            dcoord,
            ordered_labels,
            dendrogram_dict["leaves"],
            threshold_vertical,
        )
