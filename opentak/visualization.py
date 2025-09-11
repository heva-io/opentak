from __future__ import annotations

import copy
import itertools
import json
import math
import string
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import PIL
import plotly.express as px
import plotly.graph_objects as go
from PIL import ImageFilter
from plotly.subplots import make_subplots

from opentak.dendrogram_utils import create_dendrogram
from opentak.tak_theme import base_template
from opentak.tak_theme.palettes import qualitative

if TYPE_CHECKING:
    from collections.abc import MutableMapping, Sequence
    from datetime import datetime

    import numpy.typing as npt
    from PIL.Image import Image

    from opentak.clustering import Tak

DEFAULT_EVENTS_COLORS = {
    "nothing": "#FFFFFF",
    "death": "#2B1D46",
    "in": "#C9C8D6",
    "start": "#F4F2F7",
    "out": "#C9C8D6",
    "after_out": "#5E5A85",
    "other": "#8F8BAA",
}

DICT_TIME_SAMPLE = {
    "day": 1,
    "week": 7,
    "month": 30.4375,
    "quarter": 91.3125,
    "year": 365.25,
}

DENDRO_AXIS_LAYOUT = {
    "mirror": False,
    "showgrid": False,
    "showline": False,
    "zeroline": False,
    "showticklabels": False,
    "ticks": "",
}


class NumpyEncoder(json.JSONEncoder):
    """Used to serialize numpy arrays before a .json export."""

    def default(self, obj):
        """Handle numpy array serialization.

        :param obj: object to serialize
        :return: serializable representation of the object
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


@dataclass
class ExportTAK:
    """Class for exporting the TAK."""

    sorted_base: list[Any]
    dict_colors: dict
    dict_id_labels: dict

    def __init__(self, sorted_array, dico_label_color, dict_label_id):
        """Initialize the ExportTAK object.

        :param sorted_array: sorted array of patient sequences
        :param dico_label_color: dictionary mapping labels to colors
        :param dict_label_id: dictionary mapping treatments to IDs
        """
        self.sorted_base = sorted_array
        self.dict_colors = dico_label_color
        self.dict_id_labels = dict_label_id


def _remove_nan_1d(arr1d: npt.NDArray) -> npt.NDArray:
    """Remove NaN values from a 1D numpy array.

    :param arr1d: 1D numpy array
    :return: array with NaN values removed
    """
    return arr1d[~np.isnan(arr1d)]


def _reduce_matrix(
    mat: npt.NDArray, expected_sizes: tuple[int, int], method: str = "mode"
) -> tuple[npt.NDArray, int, int]:
    """Reduce the matrix to an expected size.

    This method can be customized as shown in documentation.

    :param mat: matrice to reduce
    :param expected_sizes: expected size (x,y)
    :param method: aggregation method ('max', 'median' or 'mode')
    :return: matrix reduced, kernel_x_size, kernel_y_size
    """
    y_size, x_size = mat.shape[:2]
    expected_x, expected_y = expected_sizes
    if x_size <= expected_x and y_size <= expected_y:
        # if the matrix is already smaller than the desired pooling size, skip
        return mat, 1, 1

    kernel_x_size = int(x_size / expected_x)
    kernel_y_size = int(np.floor(y_size / expected_y))
    expected_y = int(np.ceil(y_size / kernel_y_size))
    expected_x = int(np.ceil(x_size / kernel_x_size))
    padded_size = (
        expected_y * kernel_y_size,
        expected_x * kernel_x_size,
        *mat.shape[2:],
    )
    padded_matrix = np.full(padded_size, np.nan)
    padded_matrix[:y_size, :x_size, ...] = mat
    new_shape = (expected_y, kernel_y_size, expected_x, kernel_x_size, *mat.shape[2:])

    if method == "max":
        result = np.nanmax(padded_matrix.reshape(new_shape), axis=(1, 3))
    elif method == "median":
        result = np.nanmedian(padded_matrix.reshape(new_shape), axis=(1, 3))
    elif method == "mode":
        # Pretty ugly, but efficient
        result = np.fromiter(
            (
                np.bincount(
                    _remove_nan_1d(
                        padded_matrix.reshape(new_shape)[i, :, j, :].ravel()
                    ).astype("int8")
                ).argmax()
                for i, j in product(range(new_shape[0]), range(new_shape[2]))
            ),
            "int8",
            expected_x * expected_y,
        ).reshape(new_shape[0], new_shape[2])
    else:
        raise NotImplementedError(
            f"Method {method} not implemented, please choose between 'max', 'median', 'mode'"
        )
    return result, kernel_x_size, kernel_y_size


class TakVisualizer:
    def __init__(
        self, tak: Tak, dico_evt_for_legend: dict[str, str] | None = None
    ) -> None:
        """Creation of the TakVisualizer object.

        :param tak: Tak object with fitted clustering results
        :param dico_evt_for_legend: the mapping for legend names of events
        """
        # Input
        self.tak = tak

        # Used to reset the visualization if needed
        self.memory_tak: list[npt.NDArray] | None = None
        self.memory_clusters: list[list[int]] | None = None

        # Params
        # Colors
        self.default_colors = dict(DEFAULT_EVENTS_COLORS)

        self.dico_label_color: dict[str, str] = {}
        self._create_dict_label_color()

        if dico_evt_for_legend is None:
            self.dico_evt_for_legend: dict[str, str] = {}
        else:
            self.dico_evt_for_legend = dico_evt_for_legend

        # Output of the visualization
        self.array_tak_viz = None
        self.axes: dict = {}
        self.sampled_patients = np.array([])

        self.dico_id_evt: dict[int, str] = {
            value: key for key, value in self.tak.dict_label_id.items()
        }

        self.set_values: set[str | int] = set()
        self.nb_patients: int | None = None
        self.base_date: datetime | None = None
        self.fig_x_axis: dict[str, Any] | None = None

        # Duration
        self.clusters_names: list[str] | None = None
        self.list_len_clusters_ordered: list[int] | None = None

        # Additional attributes set later
        self.coef_width: float = 1.0
        self.coef_length: float = 1.0
        self.current_processed_patients: npt.NDArray = np.array([])
        self.fig: go.Figure | None = None

    def _create_dict_label_color(self) -> None:
        """Create dico 'labels' -> 'colors'.

        This method assigns colors to event labels, using default colors for
        standard events and cycling through a color palette for additional events.
        """
        # Setting default colors
        self.dico_label_color.update(self.default_colors)

        # Get evts and sort them to be sure the colors are the same when you run
        # the visualization multiple times
        evt = sorted(self.tak.dict_label_id)

        # Removing colors that are already present
        evt = [x for x in evt if x not in self.dico_label_color]

        # Get colors
        color_cycle = itertools.cycle(qualitative.default)

        # Compute dictionary
        dict_color = dict(zip(evt, color_cycle, strict=False))
        dict_color.update(self.dico_label_color)

        self.dico_label_color = dict_color

    def update_colors(
        self, dict_new_colors: dict[str, str] | None = None, **kwargs: Any
    ) -> None:
        """Update the color dictionary.

        Allows the user to set colors for specific labels.

        :param dict_new_colors: Dict 'label'->'color'
        :param kwargs: You can specify new colors here, for example by using
        .update_colors(drug_A=(24,90,0))
        """
        if dict_new_colors is not None:
            self.dico_label_color.update(dict_new_colors)

        self.dico_label_color.update(kwargs)

    def split(self, list_ids: list[int]) -> None:
        """Split the TAK array and only keep list_ids IDs.

        :param list_ids: IDs to keep in the TAK
        """
        if self.memory_tak is not None:
            raise ValueError(
                "The TAK has already been split. Please reset with the reset_split() method and try again."
            )

        complete_id_list = np.concatenate(self.tak.list_ids_clusters).ravel()

        if not set(list_ids).issubset(set(complete_id_list)):
            raise ValueError(
                "The parameter 'list_id' does not match with the IDs in the TAK object"
            )

        # Save the current TAK output that will be modified
        self.memory_tak = deepcopy(self.tak.sorted_array)
        # Restrict TAK array to patients in the sub-cohort
        self.tak.sorted_array = [
            self.tak.sorted_array[cluster][
                pd.Series(self.tak.list_ids_clusters[cluster]).isin(list_ids).to_numpy()
            ]
            for cluster in range(len(self.tak.sorted_array))
        ]

        self.memory_clusters = deepcopy(self.tak.list_ids_clusters)

        self.tak.list_ids_clusters = [
            [id_patient for id_patient in original_cluster if id_patient in list_ids]
            for original_cluster in self.tak.list_ids_clusters
        ]

    def reset_split(self) -> None:
        """Reset the TAK after a split so it contains the whole initial population."""
        if self.memory_tak is not None:
            self.tak.sorted_array = self.memory_tak
            self.memory_tak = None

        if self.memory_clusters is not None:
            self.tak.list_ids_clusters = self.memory_clusters
            self.memory_clusters = None

    def _generate_plotly_color_dictionary(
        self, reverse_dico_id: dict[int, str]
    ) -> dict:
        """Generate color dictionary for Plotly plotting.

        :param reverse_dico_id: Dict id -> label
        :return: Color dictionary for Plotly figures
        """
        dico_id_color_plotly = {}

        for id_traitement in reverse_dico_id:
            label = reverse_dico_id.get(id_traitement)
            color = self.dico_label_color.get(str(label), None)

            if isinstance(color, tuple):
                r = color[0]
                g = color[1]
                b = color[2]
                color = f"rgb({r},{g},{b})"

            dico_id_color_plotly[id_traitement] = color
        return dico_id_color_plotly

    def process_visualization(
        self,
        num_cluster: int | None = None,
        soften_angles=0,
        unblurred_events: Sequence[str] | None = None,
        base_date: datetime | None = None,
        process_medoids: bool = False,
        **kwargs,
    ):
        """Transform the image into an array of sequence (with a daily timeline) of similar patients.

        :param num_cluster: cluster to process
        :param soften_angles: size of the blur filter
        :param unblurred_events: events to not consider for the blur
        :param base_date: date to be filled in to set the first date of the x-axis
        :param process_medoids: whether to use medoids instead of patients
        Keyword Arguments:
            - `agg_patients` : aggregation method ('max', 'median' or 'mode')
            - `sampling_size` : Nb pixels de la dimension maximale
            - `min_samples` : Borne min de pixels de la dimension minimale
            - `x_range`
            - `y_range`
            - `image_length`
            - `image_width`

        If no date is entered, the x-axis remains in number of years.

        """
        if not self.tak.is_fitted:
            raise ValueError(
                "TAK is not fitted, the .fit() method has to be used in order to sort the patients."
            )
        if process_medoids:
            if not hasattr(self.tak, "sorted_array_medoides"):
                raise TypeError("TAK should be a MetaTak if process_medoids is True")
            patients = (
                self.tak.sorted_array_medoides[num_cluster]
                if num_cluster is not None
                else np.concatenate(self.tak.sorted_array_medoides)
            )
        else:
            patients = (
                self.tak.sorted_array[num_cluster]
                if num_cluster is not None
                else np.concatenate(self.tak.sorted_array)
            )

        self._run_process_visualisation(
            patients, soften_angles, unblurred_events, base_date, **kwargs
        )

    def _run_process_visualisation(
        self,
        patients: npt.NDArray,
        soften_angles: int = 0,
        unblurred_events: Sequence[str] | None = None,
        base_date: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        nb_patients = len(patients)

        # A ajouter en parametre ?
        agg_patients = kwargs.get("agg_patients", "mode")

        sampling_size = kwargs.get(
            "sampling_size", 1000
        )  # Nb pixels de la dimension maximale
        min_dim = min(*patients.shape)
        min_samples = min(
            min_dim, kwargs.get("min_samples", 100)
        )  # Borne min de pixels de la dimension minimale

        x_range = kwargs.get("x_range", (0, patients.shape[1]))  # L
        y_range = kwargs.get("y_range", (0, patients.shape[0]))  # = nb_patients

        if patients.shape[1] > patients.shape[0]:  # L>N
            aspect = patients.shape[0] / patients.shape[1]  # N/L
            sampl_sizes = (
                sampling_size,
                max(int(sampling_size * aspect), min_samples),
            )  # La dimension max est le temps
        else:
            aspect = patients.shape[1] / patients.shape[0]  # L/N
            sampl_sizes = (
                max(int(sampling_size * aspect), min_samples),
                sampling_size,
            )  # La dimension max est le nb de patients

        # if sampling size > matrix, do not sample
        if sampl_sizes[0] < patients.shape[1] and sampl_sizes[1] < patients.shape[0]:
            sampled_patients, kx, ky = _reduce_matrix(
                patients, sampl_sizes, agg_patients
            )
            x = np.arange(x_range[0], x_range[1], kx) + kx / 2
            y = np.arange(y_range[0], y_range[1], ky) + ky / 2

        else:
            sampled_patients = patients
            x = np.arange(x_range[0], x_range[1])
            y = np.arange(y_range[0], y_range[1])

        set_values = set(sampled_patients.ravel())

        if len(set_values) > len(self.dico_label_color):
            raise RuntimeError(
                f"Not enough colors are specified ({len(self.dico_label_color)} vs {{len(set_values)}} needed)"
            )

        image_length = kwargs.get("image_length", len(sampled_patients))
        image_width = kwargs.get("image_width", len(sampled_patients[0]))

        image_bigsize, image_outlines = soften_angles_sample(
            sampled_patients, soften_angles, image_length, image_width
        )
        self.coef_width = len(sampled_patients[0]) / image_width
        self.coef_length = len(sampled_patients) / image_length

        # Replace unblurred_events (death, out, after out)
        # in their original place (to avoid contradictions)
        array_image_outlines = np.array(image_outlines)

        array_image_bigsize = np.array(image_bigsize)
        if unblurred_events is not None:
            for evt in unblurred_events:
                array_image_outlines = np.where(
                    (array_image_bigsize != array_image_outlines)
                    & (array_image_bigsize == self.tak.dict_label_id[evt]),
                    self.tak.dict_label_id[evt],
                    array_image_outlines,
                )
                array_image_outlines = np.where(
                    (array_image_bigsize != array_image_outlines)
                    & (array_image_outlines == self.tak.dict_label_id[evt]),
                    array_image_bigsize,
                    array_image_outlines,
                )
        self.axes = {"x": np.array(x, dtype="int"), "y": y}
        self.sampled_patients = array_image_outlines
        self.current_processed_patients = patients
        self.set_values = set_values
        self.nb_patients = nb_patients
        self.base_date = base_date

    def _process_date_axis(
        self,
        unit_as_months,
        nb_interval_pre_offset,
        nb_interval_post_offset,
        nb_months,
        nb_interval,
    ):
        if self.base_date:
            if unit_as_months:
                start_date = self.base_date + pd.DateOffset(
                    months=-nb_interval_pre_offset * nb_months
                )
                list_months = pd.date_range(
                    start=start_date, periods=nb_interval + 1, freq=f"{nb_months}ME"
                )
                ticktext = [
                    f"{start_date.day:02}-{date.month:02}-{date.year}"
                    for date in list_months
                ]
                x_title = "in months"
            else:
                ticktext = [
                    f"{self.base_date.day:02}-{self.base_date.month:02}-{year}"
                    for year in range(
                        self.base_date.year - nb_interval_pre_offset,
                        self.base_date.year + nb_interval_post_offset + 1,
                    )
                ]
                x_title = "in years"
        elif unit_as_months:
            ticktext = nb_months * np.arange(
                -nb_interval_pre_offset, nb_interval_post_offset + 1
            )
            x_title = "in months"
        else:
            ticktext = np.arange(-nb_interval_pre_offset, nb_interval_post_offset + 1)
            x_title = "in years"

        return ticktext, x_title

    def get_plot(
        self,
        plot_title: str = "Analysis of treatment lines",
        unit_as_years: bool = False,
        unit_as_months: bool = False,
        nb_months: int | None = None,
        offset: int = 0,
        add_sep: bool = False,
        dendrogram: bool = False,
        **kwargs,
    ) -> go.Figure:
        # ruff: noqa: D301
        """Allow the display of TAK with Plotly.

        !!! note
            if unit_as_years=False and unit_as_month=False, units are in TIMESTAMP \n
            if self.base_date = True, xaxes will be in years

        !!! question "Kwargs"
            - `list_len_clusters_ordered`: (list) number of patients in each cluster,
            starting from the bottom of the TAK
            - `write_annotation`: (bool) whether or not to write the annotation at the right
            of the TAK.
            Default: True
            - `clusters_names`: (list) liste des noms des clusters à associer
            à list_len_clusters_ordered.
            Default: ["A", "B", etc...]
            - `color_annotation`: (str) color of the lines and of the text at the left of
            the graph.
            Default: "#4CA094" (green-blue)
            - `width_line`: (int) width of the lines.
            Default: 3
            - `size_annotation`: (int) size of the font for annotation at the right of the
            TAK.
            Default: 10

        :param plot_title: Title of the plot (str)
        :param unit_as_years: specify whether the units are years (True) or other (False)
        :param unit_as_months: specify whether the units are months (True) or other (False)
        :param nb_months: specify every how many months do we add a xtick (ignored if unit_as_month=False)
        :param offset: specify if an offset (in TIMESTAMP) should be apply to xaxes
        :param add_sep: whether to add horizontal line on the tak or not
        :param dendrogram: whether to add a dendrogram on the side or not
        :param kwargs: options to deal with cluster sizes, names, colors
        and width of the lines and size of the annotations

        :return: Plotly figure
        """
        dico_id_color = self._generate_plotly_color_dictionary(self.dico_id_evt)
        traces = [
            go.Heatmap(
                x=self.axes["x"],
                y=self.axes["y"],
                z=np.where(self.sampled_patients == value, value, np.nan),
                hoverinfo="none",
                colorscale=[(0.0, dico_id_color[value]), (1.0, dico_id_color[value])],
                name=f"{self.dico_evt_for_legend.get(self.dico_id_evt[int(value)], self.dico_id_evt[int(value)])}",
                showscale=False,
                showlegend=True,
                autocolorscale=False,
            )
            for i, value in enumerate(self.set_values)
        ]

        xaxis = self._get_x_axis(unit_as_years, unit_as_months, nb_months, offset)

        optimal_y_ticks = NiceScale(0, self.nb_patients)

        ticktext = [
            optimal_y_ticks.nice_min + i * optimal_y_ticks.tick_spacing
            for i in range(optimal_y_ticks.max_ticks)
        ]
        tickvals = [tick // self.coef_length for tick in ticktext]

        yaxis = {
            "title": "Patients",
            "tickmode": "array",
            "tickvals": tickvals,
            "ticktext": ticktext,
            "ticklen": 5,
            "ticks": "outside",
        }

        self.fig_x_axis = xaxis

        if dendrogram:
            fig = make_subplots(rows=1, cols=2, column_widths=[0.2, 0.8])
            fig.add_traces(traces, rows=1, cols=2)
        else:
            fig = go.Figure()
            fig.add_traces(traces)

        fig.update_layout(
            title_text=f"{plot_title} ({self.nb_patients} patients)",
            template=base_template,
        )
        fig["layout"]["title"].update(x=0.5)

        # If a dendrogram is displayed, 2 subplots are displayed
        if dendrogram:
            fig.update_layout(xaxis2=xaxis, yaxis2=yaxis)

            dendro_fig, yaxis_range, threshold_vertical = self.get_dendro(
                add_sep, **kwargs
            )

            dendro_yaxis_layout = copy.deepcopy(DENDRO_AXIS_LAYOUT)
            dendro_yaxis_layout["range"] = yaxis_range

            fig.update_layout(yaxis=dendro_yaxis_layout, xaxis=DENDRO_AXIS_LAYOUT)

            for trace in dendro_fig["data"]:
                trace["showlegend"] = False
                fig.add_trace(trace, row=1, col=1)

            fig.add_vline(
                x=-threshold_vertical,
                line_width=1,
                line_dash="dot",
                line_color="rgb(0,0,255)",
                row=1,
                col=1,
            )

        # Otherwise we only display one plot
        else:
            fig.update_layout(xaxis=xaxis, yaxis=yaxis)

        self.fig = (
            self._add_sep_on_tak_fig(fig, dendrogram=dendrogram, **kwargs)
            if add_sep
            else fig
        )

        return self.fig

    def _get_x_axis(
        self,
        unit_as_years: bool = False,
        unit_as_months: bool = False,
        nb_months: int | None = None,
        offset: int = 0,
    ):
        if unit_as_years or unit_as_months or self.base_date:
            if unit_as_years and unit_as_months:
                raise ValueError(
                    "You should choose between unit_as_years and unit_as_months"
                )

            if not unit_as_months and self.base_date:
                unit_as_years = True

            if unit_as_years:
                nb_months = 12
            elif nb_months is None:
                nb_months = 6

            timestamp_max = self.axes["x"].max()

            interval = nb_months * (365.25 / 12)
            nb_interval = int(timestamp_max // interval)

            offset_in_intervals = offset / interval
            nb_interval_post_offset = int(nb_interval - offset_in_intervals)
            nb_interval_pre_offset = int(offset_in_intervals)

            list_indexes = [
                int((offset + interval * one_interval) / self.coef_width)
                for one_interval in range(
                    -nb_interval_pre_offset, nb_interval_post_offset + 1
                )
            ]

            ticktext, x_title = self._process_date_axis(
                unit_as_months,
                nb_interval_pre_offset,
                nb_interval_post_offset,
                nb_months,
                nb_interval,
            )

            xaxis = {
                "title": f"Time ({x_title})",
                "tickmode": "array",
                "tickvals": list_indexes,
                "ticktext": ticktext,
                "ticklen": 5,
                "ticks": "outside",
            }

        else:
            optimal_x_ticks = NiceScale(
                self.axes["x"][0] - offset, self.axes["x"][-1] - offset
            )

            ticktext = [
                optimal_x_ticks.nice_min + i * optimal_x_ticks.tick_spacing
                for i in range(optimal_x_ticks.max_ticks)
            ]
            tickvals = [(tick + offset) // self.coef_width for tick in ticktext]

            xaxis = {
                "title": "Time (in days)",
                "tickmode": "array",
                "tickvals": tickvals,
                "ticktext": ticktext,
                "ticklen": 5,
                "ticks": "outside",
            }
        return xaxis

    def imshow(self):
        """Allow the display of TAK with Plotly using imshow.

        TO DO : Very basic function -> needs better implantation

        :return: Plotly figure
        """
        patients = self.array_tak_viz

        return px.imshow(patients)

    def _count_evt_at_day_d(self, base_after_tak: npt.NDArray, day_d: int) -> pd.Series:
        # ruff: noqa: D205
        """At d-day, count each observation of each events on the TAK "resolution"
        (a vertical array of 1-pixel wide on the TAK image, which means Not the number
        of occurence occording to the event log but according to the TAK resolution.).

        :param base_after_tak: 2D array, patients X timestamp, could have been reduced or not
        :param day_d: the timestamp of the day.
        :return: the pd.series that counts the observation of each events
        """
        list_values = list(self.dico_id_evt.values())

        np_count_med = np.bincount(base_after_tak[:, day_d])

        dic_count_med = {
            self.dico_id_evt.get(med_id, "None"): count
            for med_id, count in enumerate(np_count_med)
        }
        serie_count_med = pd.Series(dic_count_med).reindex(list_values).fillna(0)

        serie_count_pour_med = serie_count_med / sum(serie_count_med)

        return serie_count_pour_med.sort_values(ascending=False)

    def graph_events_rep_on_tak(
        self,
        events_not_shown: list | None = None,
        events_not_in_percent: list | None = None,
        threshold_percent: float = 2,
        represented_patients_name: str = "patients involved",
    ) -> go.Figure:
        # ruff: noqa: D205
        """Generate the graph of the distribution of each event along the tak_viz,
        based on the values of the TAK array.

        :param events_not_shown: The list of events not to show on the graph.
        :param events_not_in_percent: List of events to ignore in the %age of curves. It changes the denominator
        of other curves.
        :param threshold_percent: % minimum of patients represented on the first graph
        :param represented_patients_name: Name to give to patients, default value `patients involved`
        :return: Plotly figure
        """
        if self.nb_patients is None:
            raise ValueError(
                "TAK visualization is not processed, the .process_visualization() "
                "method has to be used in order to call this function."
            )
        events_not_shown = [] if events_not_shown is None else events_not_shown
        events_not_shown = list(set(events_not_shown))
        events_not_in_percent = (
            [] if events_not_in_percent is None else events_not_in_percent
        )
        events_not_in_percent = list(set(events_not_in_percent))

        base_after_tak = self.current_processed_patients.copy()
        x = np.arange(0, len(self.tak.sorted_array[0][0]))

        y_nb_meds = [
            self._count_evt_at_day_d(base_after_tak, day_d)
            for day_d in range(base_after_tak.shape[1])
        ]

        df_base_tak = pd.DataFrame(self.tak.base)

        evt_of_interest = [
            evt for evt in df_base_tak["EVT"].unique() if evt not in events_not_shown
        ]

        # Axe des y - valeurs des courbes
        # Dico des series temporelles, avec valeur par défaut au cas où aucun patient n'a l'un des evt à ignorer
        data_lines: MutableMapping[str, list[int]] = defaultdict(
            lambda: [0] * len(y_nb_meds)
        )
        for evt in df_base_tak["EVT"].unique():
            data_lines[evt] = [count_med[evt] for count_med in y_nb_meds]

        # Calcul du nouveau dénominateur (nombre de patients traités à l'instant t) : patient à exclure à chaque
        # timestamp
        if events_not_in_percent:
            percent_patients = [
                (1 - sum(nb_pat_remove))
                for nb_pat_remove in zip(
                    *[data_lines[x] for x in events_not_in_percent], strict=True
                )
            ]
            fig = make_subplots(rows=2, cols=1, row_heights=[0.8, 0.2])
            prefix_top_graph = (
                '<b><span style="text-decoration: underline;">Top graph :</span><br>'
            )
        else:
            percent_patients = [1] * len(y_nb_meds)
            fig = make_subplots(rows=1, cols=1)
            prefix_top_graph = ""
        threshold_cent = threshold_percent / 100

        mode = "lines"
        fig.add_trace(
            go.Scatter(
                x=[len(y_nb_meds) / 2],
                y=[1],
                mode="markers",
                line={"color": "rgba(0,0,0,0)"},
                name=f"{prefix_top_graph}Events of the TAK</b><br>",
            ),
            row=1,
            col=1,
        )

        for evt in evt_of_interest:
            # Division pour chaque timestamp, par le nombre de patients en cours de suivi
            if evt not in events_not_in_percent:
                y_evt = [
                    np.nan if percent_pt < threshold_cent else percent_evt / percent_pt
                    for percent_evt, percent_pt in zip(
                        data_lines[evt], percent_patients, strict=True
                    )
                ]

                line = {"color": f"{self.dico_label_color[evt]}"}

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y_evt,
                        line=line,
                        mode=mode,
                        name=self.dico_evt_for_legend.get(evt, evt),
                    ),
                    row=1,
                    col=1,
                )

        if events_not_in_percent:
            fig.add_trace(
                go.Scatter(
                    x=[len(y_nb_meds) / 2],
                    y=[1],
                    mode="markers",
                    line={"color": "rgba(0,0,0,0)"},
                    name='<br><br><b><span style="text-decoration: underline;">Bottom graph :</span><br>% of patients '
                    "involved <br>on top graph</b><br>",
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=percent_patients,
                    line={"color": "#9491AD", "dash": "dash"},
                    name=f"% {represented_patients_name}",
                    yaxis="y2",
                    mode=mode,
                ),
                row=2,
                col=1,
            )

        fig.update_layout(
            title=f"Events distribution according to the TAK ({self.nb_patients} patients)",
            title_x=0.5,
            yaxis={
                "dtick": 0.1,
                "title": f"Percentage of patients <br> <i>(among {represented_patients_name})</i>",
            },
            yaxis2={
                "dtick": 0.25,
                "title": f"Percentage of <br> {represented_patients_name} <br> <i>(over the whole cohort)</i>",
            },
        )
        if self.fig_x_axis is not None:
            fig.update_yaxes(range=[0, 1], tickformat="0%").update_xaxes(
                self.fig_x_axis, title_standoff=0
            )
        else:
            fig.update_yaxes(range=[0, 1], tickformat="0%")
        return fig

    def get_dendro(self, add_sep, **kwargs):
        # Get clusters size if not given throught list_len_clusters_ordered
        list_len_clusters_ordered = kwargs.get("list_len_clusters_ordered")
        if list_len_clusters_ordered is None:
            list_len_clusters_ordered = [
                len(cluster) for cluster in self.tak.list_ids_clusters
            ]

        fig, yaxis_range, threshold_vertical = create_dendrogram(
            self.tak.linkage_total,
            add_sep,
            nb_clusters=len(list_len_clusters_ordered),
            orientation="right",
        )

        return fig, yaxis_range, threshold_vertical

    def _add_sep_on_tak_fig(
        self, fig: go.Figure, dendrogram: bool, **kwargs
    ) -> go.Figure:
        # ruff: noqa: D205
        """Add horizontal separation between clusters of the tak, and annotations
        on the right with the name and the number of patients on each cluster.

        !!! question "Kwargs":
        - `list_len_clusters_ordered`: (list) number of patients in each cluster,
        starting from the bottom of the TAK
        -  `write_annotation`: (bool) whether or not to write the annotation at the right
        of the TAK.
        Default: True
        - `clusters_names`: (list) liste des noms des clusters à associer
        à list_len_clusters_ordered.
        Default: ["A", "B", etc...]
        - `color_annotation`: (str) color of the lines and of the text at the left of
        the graph.
        Default: "#4CA094" (green-blue)
        - `width_line`: (int) width of the lines.
        Default: 3
        - `size_annotation`: (int) size of the font for annotation at the right of the
        TAK.
        Default: 10
        - `coef_line_extension`: (float) the line will be extended of 15% (default value) in the right direction,
        to have a more visual delimitation between clusters names
        Default: 0.15

        :param fig: fig to edit
        :param dendrogram: whether to add a dendrogram or not (true/false)
        :param kwargs: options to deal with cluster sizes, names, colors and width
        of the lines and size of the annotations
        :return: copy of fig with horizontal lines to reveal clusters + annotations
        on the clusters names and number of patient in it
        """
        fig_cop = go.Figure(fig)

        # Get clusters size if not given throught list_len_clusters_ordered
        list_len_clusters_ordered = kwargs.get("list_len_clusters_ordered")
        if list_len_clusters_ordered is None:
            list_len_clusters_ordered = [
                len(cluster) for cluster in self.tak.list_ids_clusters
            ]

        # Default paraameters
        write_annotation = kwargs.get("write_annotation", True)
        clusters_names = kwargs.get(
            "clusters_names", get_clusters_names_default(list_len_clusters_ordered)
        )
        color_annotation = kwargs.get("color_annotation", "#4CA094")
        width_line = kwargs.get("width_line", 3)
        size_annotation = kwargs.get("size_annotation", 10)
        coef_line_extension = kwargs.get("coef_line_extension", 0.15)

        # x1 : right delimitation of the TAK
        x1 = self.axes["x"][-1] / self.coef_width
        tot_sep = 0
        # step : to race the delimitation between 2 patients and not in the middle of a patient
        step = 0.5
        kwargs_line = {"line_color": color_annotation, "line_width": width_line}

        if dendrogram:
            fig_cop.add_shape(
                x0=-x1 * 0,
                x1=x1 * (1 + coef_line_extension),
                y0=tot_sep - step,
                y1=tot_sep - step,
                row=1,
                col=2,
                **kwargs_line,
            )
            xref = "x2"
        else:
            fig_cop.add_shape(
                x0=-x1 * 0,
                x1=x1 * (1 + coef_line_extension),
                y0=tot_sep - step,
                y1=tot_sep - step,
                **kwargs_line,
            )
            xref = "x"

        # For each cluster, trace an horizontal separation and write the number of patients
        for name, sep in zip(clusters_names, list_len_clusters_ordered, strict=True):
            # Determine vertical limits of the cluster
            y_pred = tot_sep / self.coef_length
            tot_sep = sep + tot_sep
            y = tot_sep / self.coef_length

            if dendrogram:
                fig_cop.add_shape(
                    x0=-x1 * 0,
                    x1=x1 * (1 + coef_line_extension),
                    y0=y - step,
                    y1=y - step,
                    row=1,
                    col=2,
                    **kwargs_line,
                )
                yref = "y2"
            else:
                fig_cop.add_shape(
                    x0=-x1 * 0,
                    x1=x1 * (1 + coef_line_extension),
                    y0=y - step,
                    y1=y - step,
                    **kwargs_line,
                )
                yref = "y"

            if write_annotation:
                # At the middle of the cluster, add its name and size
                text = f"Group {name}: {sep} patients ({sep / sum(list_len_clusters_ordered):.0%})"
                fig_cop.add_annotation(
                    x=x1,
                    y=(y_pred + y) / 2,
                    text=text,
                    showarrow=False,
                    font={"color": color_annotation, "size": size_annotation},
                    align="left",
                    valign="middle",
                    xanchor="left",
                    yanchor="middle",
                    xref=xref,
                    yref=yref,
                )

        # Put the legend on the top of the graph (under the title)
        fig_cop.update_layout(
            margin={"t": 90, "r": 20},
            title_yref="container",
            title_y=0.98,
            legend={
                "orientation": "h",
                "x": 0.9,
                "y": 1.01,
                "xanchor": "right",
                "yanchor": "bottom",
            },
        )

        # Revome y and x axes grid
        fig_cop.update_yaxes(showgrid=False)
        fig_cop.update_xaxes(showgrid=False)

        # Kept as attributes for computation of recap table by cluster
        self.clusters_names = clusters_names
        self.list_len_clusters_ordered = list_len_clusters_ordered

        return fig_cop

    def export_tak(self, path_folder: str | Path, file_name: str = "tak_result.json"):
        """Export the TAK for datashader visualization.

        :param path_folder: Folder name to store results
        :param file_name: Result file name
        """
        export_data = ExportTAK(
            self.tak.sorted_array, self.dico_label_color, self.tak.dict_label_id
        )

        path_folder = Path(path_folder)

        # Create directory if it doesn't exist
        path_folder.mkdir(parents=True, exist_ok=True)

        with (path_folder / file_name).open("w", encoding="utf-8") as outfile:
            json.dump(asdict(export_data), outfile, cls=NumpyEncoder)


def soften_angles_sample(
    base_image: npt.NDArray,
    soften_angles: int,
    image_length: int | None,
    image_width: int | None,
) -> tuple[Image, Image]:
    """Apply blur on the picture.

    :param base_image: incoming image
    :param soften_angles: size of the blur filter
    :param image_length: length of the image before the filter is applied
    :param image_width: width of the image before the filter is applied
    :return: image_bigsize: image with size (image_width, image_length)
    and image_outlines: same size and blurred
    """
    # Convert into image and resize
    image = PIL.Image.fromarray(base_image.astype(np.uint8))
    if image_width is None or image_length is None:
        raise ValueError("image_width and image_length must be defined")
    image_bigsize = image.resize(
        (image_width, int(image_length)), resample=PIL.Image.Resampling.NEAREST
    )

    # Apply blur
    image_outlines = image_bigsize.filter(ImageFilter.ModeFilter(soften_angles))

    # Returns enlarged and blurred image
    return image_bigsize, image_outlines


def get_clusters_names_default(
    list_len_clusters_ordered: list, max_nb_clusters: int = 26
) -> list:
    """Return the list of default clusters names (reversed alphabetic order, ending by A).

    :param list_len_clusters_ordered: list of the number of patients in each
    cluster, order starting at the botton of the TAK
    :param max_nb_clusters: max number of clusters before raising an error
    :return: list of default clusters names (["F", "E", "D", ...])
    """
    nb_clusters = len(list_len_clusters_ordered)
    if nb_clusters > max_nb_clusters:
        raise ValueError(
            f"The number of clusters should be lower than {max_nb_clusters}."
        )
    clusters_names_default = list(string.ascii_uppercase[:nb_clusters])
    return clusters_names_default


def nice_plotly_show(fig):
    """Display nicely a plotly graph.

    :param fig: Plotly figure
    """
    config = {"toImageButtonOptions": {"height": None, "width": None}}
    fig.show(config=config)


def add_grid_on_tak_fig(
    fig: go.Figure,
    grid: str = "xy",
    params: dict | None = None,
) -> go.Figure:
    """Add grid on the TAK heatmap in the plotly fig "fig".

    !!! note
        default params for xgrid_params and ygrid_params: line_width = 0.5, line_dash = "dot", line_color = "grey",
        opacity = 0.5

    :param fig: plotly heatmap TAK on which to add the grid
    :param grid: whether to add grid on x and/or y axes
    :param params: dict params for add_vline (grid for x), 2 keys : "x" and/or "y", and each value is a dictionary of
    parameters
    :return: copy of fig with grid
    """
    fig_cop = go.Figure(fig)

    if params is None:
        params = {}

    # Pre-compute subplot configuration
    multiple_plots = fig.layout.grid.columns is not None
    xaxis = "xaxis2" if multiple_plots else "xaxis"
    yaxis = "yaxis2" if multiple_plots else "yaxis"
    col = 2 if multiple_plots else 1

    # Define grid line functions mapping
    # python
    grid_functions = {
        "x": lambda **kwargs: fig_cop.add_vline(**kwargs),
        "y": lambda **kwargs: fig_cop.add_hline(**kwargs),
    }

    for dim in grid:
        dim_params = params.get(dim, {})
        if not isinstance(dim_params, dict):
            raise TypeError(f"params[{dim}] should be a dictionnary")

        default_params = {
            "line_width": 0.5,
            "line_dash": "dot",
            "line_color": "grey",
            "opacity": 0.5,
        }
        default_params.update(dim_params)

        # Computing max value of the TAK graph (we must exclude graphs that are not heatmaps, eg scatters
        # representing the dendogram
        val_max = max(
            [max(trace[dim]) for trace in fig_cop.data if isinstance(trace, go.Heatmap)]
        )

        # Get the appropriate grid function for this dimension
        add_grid_line = grid_functions.get(dim)
        if add_grid_line is None:
            continue

        selected_axis_dim = {"x": xaxis, "y": yaxis}
        # Add grid lines for all valid tick values
        for val in fig_cop["layout"][selected_axis_dim[dim]]["tickvals"]:
            if val <= val_max:
                line_params = default_params.copy()
                line_params[dim] = val
                line_params["col"] = col
                add_grid_line(**line_params)

    return fig_cop


class NiceScale:
    def __init__(self, minv, maxv):
        """Compute optimal parameters to set the scale.

        :param minv: Min value of the range
        :param maxv: Max value of the range
        """
        self.max_ticks = 10
        self.tick_spacing = 0
        self.lst = 10
        self.nice_min = 0
        self.nice_max = 0
        self.min_point = minv
        self.max_point = maxv

        self.calculate()

    def calculate(self):
        """Calculate and updates values for tick spacing
        and nice minimum and maximum data points on the axis.
        """
        self.lst = self.nice_num(self.max_point - self.min_point, False)
        self.tick_spacing = self.nice_num(self.lst / (self.max_ticks - 1), True)
        self.nice_min = (
            math.floor(self.min_point / self.tick_spacing) * self.tick_spacing
        )
        self.nice_max = (
            math.ceil(self.max_point / self.tick_spacing) * self.tick_spacing
        )

    def nice_num(self, lst, rround):
        """Return a "nice" number approximately equal to range.

        :param rround: Rounds the number if rround = true
        Takes the ceiling if rround = false.
        :param lst: Local range
        """
        self.lst = lst

        exponent = math.floor(math.log10(self.lst))
        fraction = self.lst / math.pow(10, exponent)

        if rround:
            if fraction < 1.5:
                nice_fraction = 1
            elif fraction < 3:
                nice_fraction = 2
            elif fraction < 7:
                nice_fraction = 5
            else:
                nice_fraction = 10
        # ruff: noqa: PLR5501
        else:
            if fraction <= 1:
                nice_fraction = 1
            elif fraction <= 2:
                nice_fraction = 2
            elif fraction <= 5:
                nice_fraction = 5
            else:
                nice_fraction = 10

        return nice_fraction * math.pow(10, exponent)
