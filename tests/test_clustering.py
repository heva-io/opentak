import re
from unittest import result
from unittest.mock import patch
import json
import numpy as np
import pandas as pd
import pytest
from more_itertools import flatten

from opentak import TakBuilder

col = ["ID_PATIENT", "TIMESTAMP", "EVT"]
RANDOM_STATE = 42

# base test containing 2 clusters:
# cluster 1 : patients 0 and 4
# cluster 2 : patients 1, 2, 3 and 5
base = pd.DataFrame(
    [
        [0, 0, "in"],
        [0, 1, "A"],
        [0, 2, "B"],
        [0, 4, "out"],
        [1, 0, "in"],
        [1, 1, "A"],
        [1, 3, "A"],
        [1, 7, "out"],
        [2, 0, "in"],
        [2, 1, "A"],
        [2, 4, "A"],
        [2, 8, "out"],
        [3, 0, "in"],
        [3, 1, "A"],
        [3, 5, "A"],
        [3, 9, "out"],
        [4, 0, "in"],
        [4, 1, "A"],
        [4, 3, "B"],
        [4, 4, "out"],
        [6, 0, "in"],
        [6, 1, "A"],
        [6, 6, "A"],
        [6, 10, "out"],
    ],
    columns=col,
)


def test_uncomputed_pdist_ok():
    # Given
    tak = TakBuilder(base).build()
    # When
    tak.fit()
    # Then


@pytest.mark.parametrize(
    ("n_clusters, optimal_ordering"),
    [(1, True), (1, False), (2, True), (2, False), (3, True), (3, False)],
)
def test_tak(n_clusters, optimal_ordering):
    # Given
    tak = TakBuilder(base).build()
    # When
    tak.fit(n_clusters=n_clusters, optimal_ordering=optimal_ordering)
    # Then
    assert len(tak.sorted_array) == n_clusters


def test_tak_clusters_sum_ids_patients():
    # Given
    tak = TakBuilder(base).build()
    # When
    tak.fit(n_clusters=2)
    # Then
    assert sum([len(array) for array in tak.sorted_array]) == base.ID_PATIENT.nunique()
    assert set(flatten(tak.list_ids_clusters)) == set(base.ID_PATIENT.unique())


sorted_array_expected = [
    np.array(
        [
            [1, 6, 7, 7, 2, 2, 2, 2, 2, 2],
            [1, 6, 6, 7, 2, 2, 2, 2, 2, 2],
        ]
    ),
    np.array(
        [
            [1, 6, 6, 6, 6, 6, 6, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        ]
    ),
]


def test_tak_2clusters_order():
    # Given
    tak = TakBuilder(base).build(kind="hca")
    # When
    tak.fit(n_clusters=2, optimal_ordering=True)
    # Then
    for act, exp in zip(tak.sorted_array, sorted_array_expected):
        assert act.shape == exp.shape
        np.testing.assert_array_equal(act, exp)


def test_golden_test():
    """
    Test TAK clustering pipeline on a synthetic dataset with realistic volumetry
    """
    n_clusters = 4
    base = pd.read_csv("./data/golden_test_event_log_2000pat.csv")
    base_out = pd.DataFrame(
        {"ID_PATIENT": base["ID_PATIENT"].unique(), "TIMESTAMP": 10, "EVT": "out"}
    )
    base = pd.concat([base, base_out])
    tak = TakBuilder(base).build(kind="hca")
    tak.fit(n_clusters=n_clusters)
    result = np.concatenate(tak.sorted_array)

    with open(
        "./data/golden_test_sorted_array_result.json", "r", encoding="utf-8"
    ) as f:
        loaded_lists = json.load(f)

    result_expected = [np.array(lst, dtype=np.uint8) for lst in loaded_lists][0]

    with open("./data/golden_test_list_ids_patients.json", "r", encoding="utf-8") as f:
        loaded_lists_patients = json.load(f)

    # Check that all the patients sequences are ordered in the same way in the final array
    assert np.array_equal(result, result_expected)

    # Check that the patients have been divided into the correct number of clusters
    assert len(tak.list_ids_clusters) == n_clusters

    # Check that the patients are in the correct clusters and in the correct order
    assert tak.list_ids_clusters == loaded_lists_patients
