import pandas as pd
import pytest

import opentak.utils_events.utils as utils

COLS = ["ID_PATIENT", "TIMESTAMP", "EVT"]


def test_regroup_evt():
    # Given
    log = pd.DataFrame(
        [
            [1, 2, "A"],
            [1, 2, "B"],
            [1, 6, "A"],
            [2, 2, "B"],
            [2, 2, "A"],
            [2, 4, "B"],
            [2, 6, "A"],
            [2, 6, "B"],
        ],
        columns=COLS,
    )
    log_expected = pd.DataFrame(
        [[1, 2, "A + B"], [1, 6, "A"], [2, 2, "A + B"], [2, 4, "B"], [2, 6, "A + B"]],
        columns=COLS,
    )

    # When
    log_result = utils.regroup_evt(log)
    # Then
    pd.testing.assert_frame_equal(log_result, log_expected)


# Test 3 events at time 0 with 'in' included in list_ignored_evt
def test_regroup_evt_3_evts_in_included():
    # Given
    base_in_A_B_at_0 = pd.DataFrame(
        [[1, 0, "in"], [1, 0, "A"], [1, 0, "B"], [1, 2, "A"], [1, 2, "B"], [1, 3, "A"]],
        columns=COLS,
    )
    base_in_A_B_at_0_theorique = pd.DataFrame(
        [[1, 0, "in"], [1, 0, "A + B"], [1, 2, "A + B"], [1, 3, "A"]],
        columns=COLS,
    )

    # When
    base_res = utils.regroup_evt(base_in_A_B_at_0, list_ignored_evt=["in"])

    # Then
    pd.testing.assert_frame_equal(base_res, base_in_A_B_at_0_theorique)


# Test: 3 events at time 0, and ['in'] in list_ignored_evt but not in the eventlog
def test_regroup_evt_in_not_in_base_but_in_list_ignored_evt():
    # Given
    base_A_B_C_at_0 = pd.DataFrame(
        [[1, 0, "C"], [1, 0, "A"], [1, 0, "B"], [1, 2, "A"], [1, 2, "B"]],
        columns=COLS,
    )
    base_A_B_C_at_0_theorique = pd.DataFrame(
        [[1, 0, "A + B + C"], [1, 2, "A + B"]],
        columns=COLS,
    )

    # When
    base_res = utils.regroup_evt(base_A_B_C_at_0, list_ignored_evt=["in"])

    # Then
    pd.testing.assert_frame_equal(base_res, base_A_B_C_at_0_theorique)


# Test: "in" occurring at the same time as one med and "out" at the same time as two meds
def test_order_in_first_out_last():
    # Given
    log = pd.DataFrame(
        [[1, 2, "B"], [1, 2, "in"], [1, 6, "A"], [1, 6, "out"], [1, 6, "B"]],
        columns=COLS,
    )
    log_expected = log.reindex([1, 0, 2, 4, 3]).reset_index(drop=True)

    # When
    log_reorder = utils.order_in_first_out_last(log)

    # Then
    pd.testing.assert_frame_equal(log_reorder, log_expected)


# If "in" occurs after the patient's first event, leave it as-is (the checks will report the problem)
def test_order_in_first_out_last_2():
    # Given
    log = pd.DataFrame(
        [[1, 2, "B"], [1, 3, "in"], [1, 6, "A"], [1, 6, "out"], [1, 6, "B"]],
        columns=COLS,
    )
    log_expected = log.reindex([0, 1, 2, 4, 3]).reset_index(drop=True)

    # When
    log_reorder = utils.order_in_first_out_last(log)

    # Then
    pd.testing.assert_frame_equal(log_reorder, log_expected)


# When there is no "in"
def test_order_in_first_out_last_no_in():
    # Given
    log = pd.DataFrame(
        [[1, 2, "B"], [1, 6, "out"], [1, 6, "B"]],
        columns=COLS,
    )
    log_expected = log.reindex([0, 2, 1]).reset_index(drop=True)

    # When
    log_reorder = utils.order_in_first_out_last(log)

    # Then
    pd.testing.assert_frame_equal(log_reorder, log_expected)


# When there is neither "in" nor "out"
def test_order_in_first_out_last_no_in_no_out():
    # Given
    log = pd.DataFrame(
        [[1, 2, "B"], [1, 6, "A"], [1, 6, "B"]],
        columns=COLS,
    )
    log_expected = log

    # When
    log_reorder = utils.order_in_first_out_last(log)

    # Then
    pd.testing.assert_frame_equal(log_reorder, log_expected)


# When there is no "in" and it's not sorted
def test_order_in_first_out_last_no_in_not_sorted_inside():
    # Given
    log = pd.DataFrame(
        [[1, "B", 3], [1, "B", 2], [1, "out", 6], [1, "B", 6]],
        columns=["ID_PATIENT", "EVT", "TIMESTAMP"],
    )
    log_expected = log.reindex([1, 0, 3, 2]).reset_index(drop=True)

    # When
    log_reorder = utils.order_in_first_out_last(log)

    # Then
    pd.testing.assert_frame_equal(log_reorder, log_expected)


def test_get_evt_log_with_frosenset_evt():
    # Given
    log = pd.DataFrame(
        [[1, "A", 2], [1, "B", 3], [1, "A", 6], [1, "B", 6], [2, "B", 3]],
        columns=["ID_PATIENT", "EVT", "TIMESTAMP"],
    )
    log_expected = pd.DataFrame(
        [[1, frozenset({"A", "B"}), 6]], columns=["ID_PATIENT", "EVT", "TIMESTAMP"]
    ).set_index(["ID_PATIENT", "TIMESTAMP"])
    # When
    log_result = utils.get_evt_log_with_frosenset_evt(log).to_frame()
    # Then
    pd.testing.assert_frame_equal(log_result, log_expected)


log_X_A_BA_X = pd.DataFrame(
    [[1, "X", 0], [1, "A", 2], [1, "B", 3], [1, "A", 3], [1, "X", 6]],
    columns=["ID_PATIENT", "EVT", "TIMESTAMP"],
)
log_X_A_AB_X = pd.DataFrame(
    [[1, "X", 0], [1, "A", 2], [1, "A", 3], [1, "B", 3], [1, "X", 6]],
    columns=["ID_PATIENT", "EVT", "TIMESTAMP"],
)
log_A_BA = log_X_A_BA_X.query("EVT != 'X'").reset_index(drop=True)
log_A_AB = log_X_A_AB_X.query("EVT != 'X'").reset_index(drop=True)

log_X_BA_B_X = pd.DataFrame(
    [[1, "X", 0], [1, "B", 3], [1, "A", 3], [1, "B", 5], [1, "X", 6]],
    columns=["ID_PATIENT", "EVT", "TIMESTAMP"],
)
log_X_AB_B_X = pd.DataFrame(
    [[1, "X", 0], [1, "A", 3], [1, "B", 3], [1, "B", 5], [1, "X", 6]],
    columns=["ID_PATIENT", "EVT", "TIMESTAMP"],
)
log_BA_B = log_X_BA_B_X.query("EVT != 'X'").reset_index(drop=True)
log_AB_B = log_X_AB_B_X.query("EVT != 'X'").reset_index(drop=True)


@pytest.mark.parametrize(
    "log, log_expected",
    [
        # B and A should be inverted, because the following EVT is B
        (log_X_BA_B_X, log_X_AB_B_X),
        # if right order, it should stay the same
        (log_X_AB_B_X, log_X_AB_B_X),
        # B and A should be inverted, because the previous EVT is A
        (log_X_A_BA_X, log_X_A_AB_X),
        # if right order, it should stay the same
        (log_X_A_AB_X, log_X_A_AB_X),
        # same tests but where A or B is at the border of the evtlog
        (log_BA_B, log_AB_B),
        (log_AB_B, log_AB_B),
        (log_A_BA, log_A_AB),
        (log_A_AB, log_A_AB),
    ],
)
def test_deal_with_double_delivrance(log, log_expected):
    # Given : log and log_expected in params

    # When
    log_result = utils.deal_with_double_delivrance(log, reset_index=True)
    # Then
    pd.testing.assert_frame_equal(log_result, log_expected)
