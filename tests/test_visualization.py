import numpy as np
import pandas as pd
import pytest
from copy import deepcopy

import opentak.visualization as viz
from opentak import TakBuilder, TakVisualizer
from opentak.generation_cohort_tak import GenerateCohortTAK

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

NB_PATIENTS = 20
nb_days_end = 50

# Example TAK
base = GenerateCohortTAK(
    nb_patients=NB_PATIENTS, nb_days_end=nb_days_end, random_state=RANDOM_STATE
)
base.initialisation_dataframe(
    treatment_name="A",
    dose_mean=int(nb_days_end / 10),
    dose_std=int(nb_days_end / 25),
)
base.add_switch_gaussien("B")

base = base.add_in_out()

base = base.sort_values(by=["ID_PATIENT", "TIMESTAMP"])

# Objects used to build visualization
tak = TakBuilder(base).build()
tak.fit()
array_tak = tak.sorted_array

# Example TAK 2 (nb_days_end higher to test unit_as_years)

NB_PATIENTS_2 = 20
nb_days_end_2 = 850

base_2 = GenerateCohortTAK(nb_patients=NB_PATIENTS_2, nb_days_end=nb_days_end_2)
base_2.initialisation_dataframe()
base_2 = base_2.add_in_out()
base_2 = base_2.sort_values(by=["ID_PATIENT", "TIMESTAMP"])

# Objects used to build visualization
tak_2 = TakBuilder(base_2).build()
tak_2.fit()

# Test add switch linear and add drug holidays
base_3 = GenerateCohortTAK(
    nb_patients=NB_PATIENTS_2, nb_days_end=nb_days_end_2, random_state=RANDOM_STATE
)
base_3.initialisation_dataframe()
base_3.add_switch_linear(treatment_name="B")
base_3.add_drug_holidays()


def test_create_dico_label_color():
    # Given

    # When
    tak_viz = TakVisualizer(tak)
    # Then
    assert set(tak_viz.dico_label_color.keys()) == {
        "A",
        "B",
        "start",
        "in",
        "out",
        "death",
        "end",
        "nothing",
        "after_out",
        "other",
    }


col = ["ID_PATIENT", "TIMESTAMP", "EVT"]

base_log = pd.DataFrame(
    [
        [0, 0, "in"],
        [0, 1, "A"],
        [0, 2, "B"],
        [0, 3, "out"],
        [1, 0, "in"],
        [1, 1, "A"],
        [1, 4, "A"],
        [1, 5, "out"],
    ],
    columns=col,
)


def test_tak_not_fitted():
    # Given
    tak_unfitted = TakBuilder(base_log).build()
    tak_fitted = TakBuilder(base_log).build().fit()
    # When
    tak_viz_unfitted = TakVisualizer(tak_unfitted)
    tak_viz_fitted = TakVisualizer(tak_fitted)
    # Then
    with pytest.raises(ValueError, match="TAK is not fitted"):
        tak_viz_unfitted.process_visualization()

    tak_viz_fitted.process_visualization()


def test_agg_patients_notimplemented():
    # Given
    tak_viz = TakVisualizer(tak_2)
    # When / then
    with pytest.raises(
        NotImplementedError,
        match="not implemented, please choose between 'max', 'median', 'mode'",
    ):
        tak_viz.process_visualization(
            sampling_size=3, min_samples=3, agg_patients="mean"
        )


def test_update_dict_colors():
    # Given
    tak_viz = TakVisualizer.__new__(TakVisualizer)
    tak_viz.dico_label_color = {"first": "#000"}
    arg_dict_color = {"second": "#000"}
    kwarg_dict_color = {"third": "#000"}
    # When
    tak_viz.update_colors(arg_dict_color, **kwarg_dict_color)
    # Then
    assert set(tak_viz.dico_label_color) == {"first", "second", "third"}


def test_dico_evt_for_legend():
    # Given
    tak = TakBuilder(base_log).build().fit()
    tak_viz = TakVisualizer(tak, dico_evt_for_legend={"A": "Nouveau nom pour A"})
    tak_viz.process_visualization()
    # When
    figplotly = tak_viz.get_plot()
    list_names = [heat["name"] for heat in figplotly.data]
    # Then
    assert "Nouveau nom pour A" in list_names


dim_params = {"line_width": 1, "opacity": 1}


@pytest.mark.parametrize(
    "params, grid",
    [
        [None, "xy"],
        [None, "x"],
        [{"x": dim_params, "y": dim_params}, "xy"],
        [{"x": dim_params, "y": dim_params}, "x"],
        [{"x": dim_params}, "xy"],
        [{"x": {"opacity": 1}, "y": dim_params}, "xy"],
    ],
)
def test_grid_on_tak(params, grid):
    # Given
    tak = TakBuilder(base_log).build().fit()
    tak_viz = TakVisualizer(tak)
    tak_viz.process_visualization()
    figplotly = tak_viz.get_plot()

    # When
    # add grid
    figplotly = viz.add_grid_on_tak_fig(
        figplotly,
        grid=grid,
        params=params,
    )
    # Then


base_log_curves = pd.DataFrame(
    [
        [0, 0, "in"],
        [0, 1, "A"],
        [0, 3, "B"],
        [0, 5, "A"],
        [0, 9, "B"],
        [0, 12, "out"],
        [1, 0, "in"],
        [1, 1, "A"],
        [1, 4, "A"],
        [1, 5, "out"],
    ],
    columns=col,
)
tak_fitted_curves = TakBuilder(base_log_curves).build().fit()


def test_graph_events_rep_on_tak_not_processed():
    # Given
    tak_viz_curves = TakVisualizer(tak_fitted_curves)

    # When
    # Then
    with pytest.raises(ValueError, match="TAK visualization is not processed"):
        tak_viz_curves.graph_events_rep_on_tak()


def test_graph_events_rep_on_tak_traces():
    # Given
    tak_viz_curves = TakVisualizer(tak_fitted_curves)
    tak_viz_curves.process_visualization()

    # When
    fig_complete = tak_viz_curves.graph_events_rep_on_tak()
    fig_a_b = tak_viz_curves.graph_events_rep_on_tak(
        events_not_shown=["start", "end", "in", "out"],
        events_not_in_percent=["start", "end", "in", "out"],
    )
    # Then
    assert len(fig_complete.data) == 7
    assert len(fig_a_b.data) == 5


def test_graph_events_rep_on_tak_values():
    # Given
    tak_viz_curves = TakVisualizer(tak_fitted_curves)
    tak_viz_curves.process_visualization()
    percent_a = [0, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0, 0, 0]

    # When
    fig = tak_viz_curves.graph_events_rep_on_tak(
        events_not_shown=["start", "end", "in", "out", "B"]
    )
    # Then
    assert list(fig.data[1]["y"]) == percent_a


def test_graph_events_rep_on_tak_percent_patients():
    # Given
    tak_viz_curves = TakVisualizer(tak_fitted_curves)
    tak_viz_curves.process_visualization()
    percent_patients = [1.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0, 1.0]

    # When
    fig = tak_viz_curves.graph_events_rep_on_tak(events_not_in_percent=["A"])
    # Then
    assert list(fig.data[-1]["y"]) == percent_patients


def test_graph_events_rep_on_tak_xaxes():
    # Given
    tak_viz_curves = TakVisualizer(tak_fitted_curves)
    tak_viz_curves.process_visualization()
    len_xaxes = 12
    # When
    fig = tak_viz_curves.graph_events_rep_on_tak()
    # Then
    assert len(fig.data[-1]["x"]) == len_xaxes


def test_graph_events_rep_on_tak_xaxes_with_coef_width():
    # Given
    image_width = 6
    tak_viz_curves = TakVisualizer(tak_fitted_curves)
    tak_viz_curves.process_visualization(image_width=image_width)
    figplotly = tak_viz_curves.get_plot()
    # When
    fig = tak_viz_curves.graph_events_rep_on_tak()
    # Then
    assert all(fig.data[-1]["x"] == figplotly.data[-1]["x"])


def test_get_sub_clusters_names_default():
    # Given
    list_len_clusters_ordered = [2, 5, 3, 6]
    # When
    sub_clusters_names_default = viz.get_clusters_names_default(
        list_len_clusters_ordered
    )
    # Then
    assert sub_clusters_names_default == ["A", "B", "C", "D"]


base_log_subclusters = pd.DataFrame(
    [
        [0, 0, "in"],
        [0, 1, "A"],
        [0, 2, "B"],
        [0, 3, "out"],
        [1, 0, "in"],
        [1, 1, "A"],
        [1, 4, "A"],
        [1, 5, "out"],
        [2, 0, "in"],
        [2, 1, "A"],
        [2, 4, "A"],
        [2, 6, "out"],
        [3, 0, "in"],
        [3, 1, "A"],
        [3, 5, "A"],
        [3, 6, "out"],
    ],
    columns=col,
)


@pytest.mark.parametrize(("method"), ["mode", "median", "max"])
def test_sampling(method):
    # test that the sampling works when samplig_size < matrix_size
    # the objective is to call reduce matrix with methode=mode

    # Given
    # base of 8 patients t be sure to shrink it
    base = pd.concat(
        [
            base_log_subclusters,
            base_log_subclusters.assign(
                ID_PATIENT=base_log_subclusters["ID_PATIENT"] + 4
            ),
        ],
        axis=0,
    )
    tak = TakBuilder(base).build()
    tak.fit(n_clusters=1)
    tak_viz = TakVisualizer(tak)
    tak_viz.process_visualization(
        **{"sampling_size": 2, "min_samples": 2, "agg_patients": method}
    )
    # When
    fig_with_sep = tak_viz.get_plot(add_sep=True)
    # Then


def test_nb_shapes_in_fig():
    # Given
    n_clusters = 3
    tak = TakBuilder(base_log_subclusters).build()
    tak.fit(n_clusters=n_clusters)
    tak_viz = TakVisualizer(tak)
    tak_viz.process_visualization()
    # When
    fig_with_sep = tak_viz.get_plot(add_sep=True)
    # Then
    assert len(fig_with_sep.layout.annotations) == n_clusters
    assert len(fig_with_sep.layout.shapes) == n_clusters + 1


def test_xaxis():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization()
    # When
    fig = tak_viz.get_plot()
    # Then
    assert fig.layout.xaxis.ticktext[0] == 0
    assert fig.layout.xaxis.tickvals[0] == 0


def test_xaxis_offset():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization()
    # When
    fig = tak_viz.get_plot(offset=100)
    # Then
    assert fig.layout.xaxis.ticktext[0] == -100
    assert fig.layout.xaxis.tickvals[0] == 0


def test_xaxis_base_date():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization(base_date=pd.to_datetime("2010-01-04"))
    # When
    fig = tak_viz.get_plot()
    # Then
    assert fig.layout.xaxis.ticktext[0] == "04-01-2010"
    assert fig.layout.xaxis.tickvals[0] == 0


@pytest.mark.parametrize(
    "offset, first_ticktext, first_tickval",
    [
        (366, "04-01-2009", 0),
        (364, "04-01-2010", 364),
    ],
)
def test_xaxis_base_date_offset(offset, first_ticktext, first_tickval):
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization(base_date=pd.to_datetime("2010-01-04"))
    # When
    fig = tak_viz.get_plot(offset=offset)
    # Then
    assert fig.layout.xaxis.ticktext[0] == first_ticktext
    assert fig.layout.xaxis.tickvals[0] == first_tickval


def test_xaxis_unit_as_years():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization()
    # When
    fig = tak_viz.get_plot(unit_as_years=True)
    # Then
    assert fig.layout.xaxis.ticktext[0] == 0
    assert fig.layout.xaxis.tickvals[0] == 0
    assert fig.layout.xaxis.ticktext[1] == 1
    assert fig.layout.xaxis.tickvals[1] == 365


def test_xaxis_unit_as_years_offset():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization()
    # When
    fig = tak_viz.get_plot(unit_as_years=True, offset=364)
    # Then
    assert fig.layout.xaxis.ticktext[0] == 0
    assert fig.layout.xaxis.tickvals[0] == 364
    assert fig.layout.xaxis.ticktext[1] == 1
    assert fig.layout.xaxis.tickvals[1] == 364 + 365


def test_xaxis_unit_as_years_offset_base_date():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization(base_date=pd.to_datetime("2010-01-04"))
    # When
    fig = tak_viz.get_plot(unit_as_years=True, offset=366)
    # Then
    assert fig.layout.xaxis.ticktext[0] == "04-01-2009"
    assert fig.layout.xaxis.tickvals[0] == 0
    assert fig.layout.xaxis.ticktext[1] == "04-01-2010"
    assert fig.layout.xaxis.tickvals[1] == 366


def test_xaxis_unit_as_months_default():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization()
    # When
    fig = tak_viz.get_plot(unit_as_months=True)
    # Then
    assert fig.layout.xaxis.ticktext[0] == 0
    assert fig.layout.xaxis.tickvals[0] == 0
    assert fig.layout.xaxis.ticktext[1] == 6
    assert fig.layout.xaxis.tickvals[1] == int(365 / 2)


def test_xaxis_unit_as_months_nb_months_2():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization()
    # When
    fig = tak_viz.get_plot(unit_as_months=True, nb_months=2)
    # Then
    assert fig.layout.xaxis.ticktext[0] == 0
    assert fig.layout.xaxis.tickvals[0] == 0
    assert fig.layout.xaxis.ticktext[1] == 2
    assert fig.layout.xaxis.tickvals[1] == int(365 / 6)


def test_xaxis_unit_as_months_offset():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization()
    # When
    fig = tak_viz.get_plot(unit_as_months=True, offset=366)
    # Then
    assert fig.layout.xaxis.ticktext[0] == -12
    assert fig.layout.xaxis.tickvals[0] == 0
    assert fig.layout.xaxis.ticktext[1] == -6
    assert fig.layout.xaxis.tickvals[1] == 183


def test_xaxis_unit_as_months_offset_base_date():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization(base_date=pd.to_datetime("2010-01-04"))
    # When
    fig = tak_viz.get_plot(unit_as_months=True, offset=366)
    # Then
    assert fig.layout.xaxis.ticktext[0] == "04-01-2009"
    assert fig.layout.xaxis.tickvals[0] == 0
    assert fig.layout.xaxis.ticktext[1] == "04-07-2009"
    assert fig.layout.xaxis.tickvals[1] == 183


def test_xaxis_unit_as_months_offset_base_date_image_width():
    # Given
    tak_viz = TakVisualizer(tak_2)
    tak_viz.process_visualization(
        base_date=pd.to_datetime("2010-01-04"), image_width=425
    )
    # When
    fig = tak_viz.get_plot(unit_as_months=True, offset=366)
    # Then
    assert fig.layout.xaxis.ticktext[0] == "04-01-2009"
    assert fig.layout.xaxis.tickvals[0] == 0
    assert fig.layout.xaxis.ticktext[1] == "04-07-2009"
    assert fig.layout.xaxis.tickvals[1] == int(183 / 2)


@pytest.mark.parametrize(
    "n_clusters",
    [1, 3],
)
def test_tak_split(n_clusters):
    # Given
    tak = TakBuilder(base).build()
    tak.fit(n_clusters=n_clusters)

    tak_viz = TakVisualizer(deepcopy(tak))
    tak_viz.process_visualization()

    initial_tak_values = deepcopy(tak_viz.tak.sorted_array)
    initial_cluster_values = deepcopy(tak_viz.tak.list_ids_clusters)

    # When
    ids_to_select = [0, 1, 2, 3]
    tak_viz.split(
        list_ids=ids_to_select,
    )

    # Then

    # Test filtering

    # on the whole ordered base
    assert len(np.concatenate(tak_viz.tak.sorted_array)) == len(ids_to_select)
    assert set(ids_to_select).issubset(
        set(np.concatenate(tak_viz.tak.list_ids_clusters).ravel())
    )

    # Test cache
    for cluster_number in range(len(initial_tak_values)):
        np.testing.assert_array_equal(
            tak_viz.memory_tak[cluster_number], initial_tak_values[cluster_number]
        )
        np.testing.assert_array_equal(
            tak_viz.memory_clusters[cluster_number],
            initial_cluster_values[cluster_number],
        )


# FAILED
def test_tak_split_cache():
    # Given
    tak_viz = TakVisualizer(deepcopy(tak))
    tak_viz.process_visualization()

    # When
    ids_to_select = [0, 1, 2, 3]
    tak_viz.split(
        list_ids=ids_to_select,
    )

    # Then
    with pytest.raises(ValueError):
        tak_viz.split(
            list_ids=ids_to_select,
        )


def test_tak_split_reset():
    # Given
    tak_viz = TakVisualizer(deepcopy(tak))
    tak_viz.process_visualization()

    initial_tak_values = deepcopy(tak_viz.tak.sorted_array)
    initial_cluster_values = deepcopy(tak_viz.tak.list_ids_clusters)

    # When
    ids_to_select = [0, 1, 2, 3]
    tak_viz.split(
        list_ids=ids_to_select,
    )

    tak_viz.reset_split()

    # Then
    assert tak_viz.memory_tak is None
    assert tak_viz.memory_clusters is None

    for cluster_number in range(len(initial_tak_values)):
        np.testing.assert_array_equal(
            tak_viz.tak.sorted_array[cluster_number], initial_tak_values[cluster_number]
        )
        np.testing.assert_array_equal(
            tak_viz.tak.list_ids_clusters[cluster_number],
            initial_cluster_values[cluster_number],
        )


def test_tak_split_wrong_ids():
    # Given
    tak_viz = TakVisualizer(deepcopy(tak))
    tak_viz.process_visualization()

    # When
    ids_to_select = [19, 20, 21, 22]

    # Then
    with pytest.raises(ValueError):
        tak_viz.split(
            list_ids=ids_to_select,
        )


@pytest.mark.parametrize(
    "dendrogram, result",
    [
        (True, (range(1, 2), range(1, 3))),
        (False, None),
    ],
)
def test_tak_dendrogram_option(dendrogram, result):
    """Tests if the dendrogram option adds a fig on the plotly output."""

    # Given
    tak_viz = TakVisualizer(deepcopy(tak))
    tak_viz.process_visualization()

    # When
    fig = tak_viz.get_plot(dendrogram=dendrogram)

    if dendrogram:
        assert fig._get_subplot_rows_columns() == result
    else:
        with pytest.raises(Exception):
            fig._get_subplot_rows_columns()
