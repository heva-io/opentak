import numpy as np
import pandas as pd
import pytest

import opentak.utils_events.checks as checks
import opentak.utils_events.preprocessing as prep

col = ["ID_PATIENT", "TIMESTAMP", "EVT"]

base_without_in = pd.DataFrame([[1, 0, "A"], [1, 2, "B"], [1, 5, "out"]], columns=col)
base_without_out = pd.DataFrame([[1, 0, "in"], [1, 2, "B"], [1, 5, "C"]], columns=col)
base_with_ttmt_before_in = pd.DataFrame(
    [[1, 0, "B"], [1, 2, "in"], [1, 5, "C"], [1, 8, "out"]], columns=col
)
base_with_ttmt_after_out = pd.DataFrame(
    [[1, 0, "in"], [1, 2, "B"], [1, 5, "out"], [1, 8, "C"]], columns=col
)
base_duplicated_rows = pd.DataFrame(
    [[1, 0, "in"], [1, 2, "B"], [1, 2, "B"], [1, 5, "C"], [1, 8, "out"]], columns=col
)


@pytest.mark.parametrize(
    "base, match_expected",
    [
        (base_without_in, "do not have 'in'"),
        (base_without_out, "do not have 'out'"),
        (base_with_ttmt_before_in, "'in' after their first treatment"),
        (base_with_ttmt_after_out, "'out' before their last treatment"),
        (base_duplicated_rows, "duplicate rows"),
    ],
)
def test_ValueError(base, match_expected):
    # Given

    # When
    with pytest.raises(ValueError, match=match_expected):
        checks.Checks(base)
    # Then


base = pd.DataFrame(
    [[1, 0, "in"], [1, 5, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
)
base_expected = pd.DataFrame(
    [[1, 0, "in"], [1, 2, "B"], [1, 5, "C"], [1, 8, "out"]], columns=col
)
base_first_day_ttmt = pd.DataFrame(
    [[1, 2, "B"], [1, 2, "in"], [1, 5, "C"], [1, 8, "out"]], columns=col
)
base_first_day_ttmt_expected = pd.DataFrame(
    [[1, 2, "in"], [1, 2, "B"], [1, 5, "C"], [1, 8, "out"]], columns=col
)
base_last_day_ttmt = pd.DataFrame(
    [[1, 0, "in"], [1, 2, "B"], [1, 8, "out"], [1, 8, "C"]], columns=col
)
base_last_day_ttmt_expected = pd.DataFrame(
    [[1, 0, "in"], [1, 2, "B"], [1, 8, "C"], [1, 8, "out"]], columns=col
)


@pytest.mark.parametrize(
    "base, base_expected",
    [
        (base, base_expected),
        (base_first_day_ttmt, base_first_day_ttmt_expected),
        (base_last_day_ttmt, base_last_day_ttmt_expected),
    ],
)
def test_ordonne_checks(base, base_expected):
    # Given

    # When
    base_ordonnee = checks.Checks(base).base
    # Then
    assert ((base_ordonnee == base_expected).all()).all()


base_double_delivery_same_day = pd.DataFrame(
    [[1, 0, "in"], [1, 2, "B"], [1, 2, "C"], [1, 8, "out"]], columns=col
)
base_with_ttmt_on_in = pd.DataFrame(
    [[1, 0, "in"], [1, 0, "B"], [1, 2, "C"], [1, 8, "out"]], columns=col
)
base_with_ttmt_on_out = pd.DataFrame(
    [[1, 0, "in"], [1, 2, "B"], [1, 8, "C"], [1, 8, "out"]], columns=col
)


@pytest.mark.parametrize(
    "base, multiple_delivery_expected",
    [
        (base_double_delivery_same_day, True),
        (base_with_ttmt_on_in, False),
        (base_with_ttmt_on_out, False),
    ],
)
def test_mutiple_delivrance_on_same_date(base, multiple_delivery_expected):
    # Given

    # When
    multiple_delivery = checks.Checks(base).check_mutiple_delivrance_on_same_date()
    # Then
    assert multiple_delivery == multiple_delivery_expected


def test_stable_sort():
    # Given
    base = pd.DataFrame(
        [[1, 0, "in"], [1, 5, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
    )
    expected = pd.DataFrame(
        [[1, 0, "in"], [1, 2, "B"], [1, 5, "C"], [1, 8, "out"]], columns=col
    )
    # When
    sorted_base = prep.stable_sort(base)
    # Then
    pd.testing.assert_frame_equal(sorted_base, expected)


def test_stable_mergesort():
    # Given
    base = pd.DataFrame(
        [[1, 0, "in"], [1, 0, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
    )
    expected = pd.DataFrame(
        [[1, 0, "in"], [1, 0, "C"], [1, 2, "B"], [1, 8, "out"]], columns=col
    )
    # When
    sorted_base = prep.stable_sort(base)
    # Then
    pd.testing.assert_frame_equal(sorted_base, expected)
