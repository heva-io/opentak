from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from opentak.utils_events.preprocessing import stable_sort

from opentak.clustering import TakHca
from opentak.preprocessing import TakBuilder, _validate_args

RANDOM_STATE = 42

col = ["ID_PATIENT", "TIMESTAMP", "EVT"]

base_in_at_0 = pd.DataFrame(
    [[1, 0, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
)
base_expected_in_at_0 = pd.DataFrame(
    [[1, 0, "start"], [1, 0, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"]],
    columns=col,
)
base_in_at_1 = pd.DataFrame(
    [[1, 1, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
)
base_expected_in_at_1 = pd.DataFrame(
    [[1, 0, "start"], [1, 1, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"]],
    columns=col,
)


@pytest.mark.parametrize(
    "base, base_expected",
    [(base_in_at_0, base_expected_in_at_0), (base_in_at_1, base_expected_in_at_1)],
)
def test_add_start(base, base_expected):
    # Given
    tak_builder = TakBuilder.__new__(TakBuilder)
    tak_builder.base = base
    # When
    tak_builder._add_start()
    # Then
    pd.testing.assert_frame_equal(base_expected, stable_sort(tak_builder.base))


base = pd.DataFrame(
    [[1, 0, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
)
base_expected = pd.DataFrame(
    [[1, 0, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"], [1, 28, "end"]], columns=col
)
base_out_at_nbjoursend = pd.DataFrame(
    [[1, 0, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
)
base_out_at_nbjoursend_expected = pd.DataFrame(
    [[1, 0, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "out"], [1, 8, "end"]], columns=col
)

base_death = pd.DataFrame(
    [[1, 1, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "death"]], columns=col
)
base_death_expected = pd.DataFrame(
    [[1, 1, "in"], [1, 1, "C"], [1, 2, "B"], [1, 8, "death"], [1, 12, "end"]],
    columns=col,
)


@pytest.mark.parametrize(
    "base, base_expected, max_days",
    [
        (base, base_expected, 28),
        (base_out_at_nbjoursend, base_out_at_nbjoursend_expected, 8),
        (base_death, base_death_expected, 12),
    ],
)
def test_add_end(base, base_expected, max_days):
    # Given
    tak_builder = TakBuilder.__new__(TakBuilder)
    tak_builder.base = base
    tak_builder.max_days = max_days
    # When
    tak_builder._add_end()
    # Then
    pd.testing.assert_frame_equal(base_expected, stable_sort(tak_builder.base))


base = pd.DataFrame(
    [
        [1, 0, "in", 1],
        [1, 1, "C", 1],
        [1, 2, "B", 6],
        [1, 8, "out", 2],
    ],
    columns=col + ["evt_duration"],
)
array_expected = np.array([[1, 7, 6, 6, 6, 6, 6, 6]])

base_in_on_ttmt = pd.DataFrame(
    [
        [1, 1, "in", 0],
        [1, 1, "C", 1],
        [1, 2, "B", 6],
        [1, 8, "out", 2],
    ],
    columns=col + ["evt_duration"],
)
array_in_on_ttmt_expected = np.array([[0, 7, 6, 6, 6, 6, 6, 6]])

base_plusieurs_patients = pd.DataFrame(
    [
        [1, 1, "in", 0],
        [1, 1, "C", 1],
        [1, 2, "B", 6],
        [1, 8, "out", 2],
        [2, 5, "in", 3],
        [2, 8, "B", 1],
        [2, 9, "out", 1],
    ],
    columns=col + ["evt_duration"],
)

array_plusieurs_patients_expected = np.array(
    [[0, 7, 6, 6, 6, 6, 6, 6, 2], [0, 0, 0, 0, 0, 1, 1, 1, 6]]
)

default_evts = ["start", "in", "out", "death", "end", "nothing"]


def test_create_dico_id():
    # Given
    tak_builder = TakBuilder(base_plusieurs_patients)
    # When
    tak_builder._create_dict_label_id()
    # Then

    evts_total = set(tak_builder.base["EVT"].unique())
    evts_total.update(default_evts)

    assert evts_total == set(tak_builder.dict_label_id)


def test_create_nbjours_max():
    # Given
    # When
    tak_builder = TakBuilder(base_plusieurs_patients)
    # Then
    assert tak_builder.max_days == 9


@pytest.mark.parametrize(
    "base, array_expected",
    [
        (base, array_expected),
        (base_in_on_ttmt, array_in_on_ttmt_expected),
        (base_plusieurs_patients, array_plusieurs_patients_expected),
    ],
)
def test_traitement_to_array(base, array_expected):
    # Given
    tak_builder = TakBuilder(base)
    # When
    tak_builder._create_array_from_evt_log()
    # Then
    np.testing.assert_array_equal(array_expected, tak_builder.array)


@pytest.mark.parametrize(
    "kind, expected_instance",
    [
        ("hca", TakHca),
    ],
)
def test_builder_instance(kind, expected_instance):
    # Given
    tak_builder = TakBuilder(base)
    # When
    tak = tak_builder.build(kind=kind)
    # Then
    assert isinstance(tak, expected_instance)


def test_builder_instance_ko():
    # Given
    tak_builder = TakBuilder(base)
    kind = "bad_kind_of_tak"
    # When
    with pytest.raises(ValueError):
        tak_builder.build(kind=kind)


def test_missing_column():
    # Given
    df = pd.DataFrame(dict(ID_PATIENT=[], EVT=[]))
    # When
    # Then
    with pytest.raises(ValueError, match="TIMESTAMP"):
        _validate_args(df, None)


def test_negative_max_day():
    # Given
    df = pd.DataFrame(dict(ID_PATIENT=[], EVT=[], TIMESTAMP=[]))
    max_days = -1
    # When
    # Then
    with pytest.raises(ValueError, match="positive"):
        _validate_args(df, max_days)
