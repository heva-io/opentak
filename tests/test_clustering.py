import re
from unittest.mock import patch

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
    [(1, True), (1, False), (2, True), (2, False)],
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
            [1, 6, 6, 6, 6, 6, 6, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 2],
            [1, 6, 7, 7, 2, 2, 2, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 6],
        ]
    ),
    np.array(
        [
            [1, 6, 6, 7, 2, 2, 2, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 2, 2],
        ]
    ),
]

sorted_array_expected_homogeneous_clust = [
    np.array(
        [
            [1, 6, 6, 6, 6, 6, 6, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 2],
            [1, 6, 7, 7, 2, 2, 2, 2, 2, 2],
        ]
    ),
    np.array(
        [
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [1, 6, 6, 7, 2, 2, 2, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 2, 2],
        ]
    ),
]

sorted_array_expected_single_clust = [
    np.array(
        [
            [1, 6, 6, 6, 6, 6, 6, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 2],
            [1, 6, 7, 7, 2, 2, 2, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 6, 6],
            [1, 6, 6, 7, 2, 2, 2, 2, 2, 2],
            [1, 6, 6, 6, 6, 6, 6, 6, 2, 2],
        ]
    )
]

tak_hca = TakBuilder(base).build(kind="hca")
tak_hca.fit(n_clusters=2)
