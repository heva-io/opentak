import numpy as np
import pandas as pd

from opentak.logger import logger
from opentak.utils_events.checks import Checks
from opentak.utils_events.preprocessing import stable_sort


def get_evt_log_with_frosenset_evt(
    log: pd.DataFrame,
    list_ignored_evt: list[str] | None = None,
) -> pd.Series:
    """Transform the log to group concomitant events and keep only those.

    This prepares the data for analysis by identifying events occurring at the
    same time for the same patient and replacing their set of events with a
    ``frozenset``.

    :param log: DataFrame with columns "ID_PATIENT", "EVT" and "TIMESTAMP"
    :param list_ignored_evt: events that should never be considered concomitant
        (e.g., 'in', 'out', 'death')
    :return: Series of only concomitant events (same patient and timestamp) where
        values are the ``frozenset`` of events
    """
    if list_ignored_evt is not None:
        log = log.query("EVT not in @list_ignored_evt")

    log_duplicated = log[log.duplicated(subset=["ID_PATIENT", "TIMESTAMP"], keep=False)]

    log_duplicated_grouped = log_duplicated.groupby(["ID_PATIENT", "TIMESTAMP"])[
        "EVT"
    ].agg(frozenset)

    return log_duplicated_grouped


def regroup_evt(
    log: pd.DataFrame,
    list_ignored_evt: list[str] | None = None,
    list_fzset_to_be_regrouped: list | None = None,
    sep: str = " + ",
) -> pd.DataFrame:
    """Group concomitant events into a single event label like "A + B".

    :param log: DataFrame with columns "ID_PATIENT", "EVT" and "TIMESTAMP"
    :param list_ignored_evt: events that should never be considered concomitant
        (e.g., 'in', 'out', 'death')
    :param list_fzset_to_be_regrouped: list of frozensets describing which
        combinations to group (e.g., [frozenset({"A", "B"}),
        frozenset({"B", "D"})]). If None, all concomitant events are grouped.
    :param sep: separator used when concatenating events (e.g., sep=" + "
        turns events "A" and "B" into the single event label "A + B")
    :return: DataFrame where targeted concomitant events belonging to list_fzset_to_be_regrouped are grouped into a
        single event label
    """
    log_cop = log.copy()

    frozen_set = pd.DataFrame(get_evt_log_with_frosenset_evt(log_cop, list_ignored_evt))

    if list_fzset_to_be_regrouped:
        if list_ignored_evt:
            elmt_of_list_fzset_to_be_regrouped = {
                x for fzset in list_fzset_to_be_regrouped for x in fzset
            }
            list_elmt_pb = elmt_of_list_fzset_to_be_regrouped & set(list_ignored_evt)
            if len(list_elmt_pb):
                logger.warning(
                    "Some elements from list_ignored_evt appear in list_fzset_to_be_regrouped; they will be ignored (%s)",
                    list_elmt_pb,
                )

        frozen_set = frozen_set[frozen_set["EVT"].isin(list_fzset_to_be_regrouped)]

    if len(frozen_set):
        if list_ignored_evt is None:
            list_ignored_evt = []
        log_ignored = log_cop[log_cop["EVT"].isin(list_ignored_evt)].set_index(
            ["ID_PATIENT", "TIMESTAMP"]
        )
        log_not_ignored = log_cop[~log_cop["EVT"].isin(list_ignored_evt)].set_index(
            ["ID_PATIENT", "TIMESTAMP"]
        )

        frozen_set["EVT"] = frozen_set["EVT"].apply(list).apply(sorted).apply(sep.join)
        log_not_ignored.loc[frozen_set.index, "EVT"] = frozen_set["EVT"]

        log_tot = (
            pd.concat([log_ignored, log_not_ignored])
            .reset_index()
            .drop_duplicates(subset=["EVT", "TIMESTAMP", "ID_PATIENT"])
            .reset_index(drop=True)
        )

        log_tot = order_in_first_out_last(log_tot)

    else:
        log_tot = log_cop

    return log_tot


def order_in_first_out_last(log: pd.DataFrame) -> pd.DataFrame:
    """Reorder log according to ID_PATIENT and TIMESTAMP, with in first and out last when several EVT.

    :param log: df to be reordored
    :return: log reordored
    """
    log_in = log[log["EVT"].eq("in")]
    log_out = log[log["EVT"].eq("out")]
    log_other = log[~log["EVT"].isin(["in", "out"])].sort_values(
        ["ID_PATIENT", "TIMESTAMP"]
    )
    log_reorder = stable_sort(pd.concat([log_in, log_other, log_out]))

    return log_reorder


def deal_with_double_delivrance(base, reset_index=True) -> pd.DataFrame:
    """Swap the order of rows in an event log for specific double-delivery cases.

    When a patient has a treatment subsequence like A-B-A with B and A given at
    the same ``TIMESTAMP``, the B-A pair at that timestamp is swapped to A-B.
    Likewise for A-B-A where A-B are given at the same timestamp: the pair is
    reordered to A-A then B at the next timestamp, to match the treatment logic.

    :param base: event log with columns ID_PATIENT, TIMESTAMP and EVT
    :param reset_index: reset the index before returning (rows may be permuted)

    !!! note

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|y|B|
        |10|y|A|
        |...|...|...|

        with x < y, becomes

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|y|A|
        |10|y|B|
        |...|...|...|

        and

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|x|B|
        |10|y|A|
        |...|...|...|

        with x < y, becomes

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|x|A|
        |10|y|B|
        |...|...|...|

    :return: event log where rows were reordered to reflect the patient's treatment logic
    """
    base_cop = base.copy()
    try:
        pd.testing.assert_frame_equal(
            base_cop,
            base_cop.sort_values(["ID_PATIENT", "TIMESTAMP"], kind="mergesort"),
        )
    except AssertionError:
        logger.warning(
            "The base was not sorted by 'ID_PATIENT' and 'TIMESTAMP', it has been sorted \
                       To silence the warning, use base.sort_values(['ID_PATIENT', 'TIMESTAMP'], kind='mergesort')\
                        before this function"
        )

    conditions = (
        base_cop["ID_PATIENT"].eq(base_cop["ID_PATIENT"].shift(-1))
        & base_cop["TIMESTAMP"].eq(base_cop["TIMESTAMP"].shift(-1))
        & base_cop["EVT"].ne(base_cop["EVT"].shift(-1))
        & (
            base_cop["EVT"].eq(base_cop["EVT"].shift(-2))
            | base_cop["EVT"].shift(+1).eq(base_cop["EVT"].shift(-1))
        )
        & (
            (base_cop["EVT"] != base_cop["EVT"].shift(+1))
            & (base_cop["EVT"].shift(-1) != (base_cop["EVT"].shift(-2)))
        )
    )

    conditions = conditions.reset_index(name="ttmt_to_move_down")
    conditions["new_index"] = conditions["index"]
    conditions["ttmt_to_move_up"] = (
        conditions["ttmt_to_move_down"].shift(+1).fillna(False)
    )
    conditions.loc[conditions["ttmt_to_move_down"], "new_index"] = conditions.loc[
        conditions["ttmt_to_move_up"], "index"
    ].to_numpy()
    conditions.loc[conditions["ttmt_to_move_up"], "new_index"] = conditions.loc[
        conditions["ttmt_to_move_down"], "index"
    ].to_numpy()
    base_cop = base_cop.reindex(conditions["new_index"].values)

    if reset_index:
        base_cop = base_cop.reset_index(drop=True)

    return base_cop


def add_evt_duration(base: pd.DataFrame, check=True) -> pd.DataFrame:
    """Add a column with the delay until the next event (treatment duration).

    :param base: DataFrame — one row per event for a patient; columns:
        'ID_PATIENT', 'EVT', 'TIMESTAMP'
    :param check: set to False to skip validations (e.g., if there is neither
        'in' nor 'out')
    :return: copy of the input with an additional ``evt_duration`` column

    .. note:: Regardless of ``check`` being True or False, the dataset is
        reordered because it is required for correct computation.
    """
    base_copy = _check_or_not(base, check)

    base_copy["evt_duration"] = base_copy["TIMESTAMP"].shift(-1).diff().copy()

    base_copy["evt_duration"].iloc[0] = (
        base_copy["TIMESTAMP"].iloc[1] - base_copy["TIMESTAMP"].iloc[0]
    )

    conditions = (
        (base_copy["EVT"].shift(+1) != "in")
        & (base_copy["EVT"].shift(+1) != "start")
        & base_copy["evt_duration"].shift(+1).eq(0)
        & (base_copy["EVT"] != "out")
        & (base_copy["EVT"] != "end")
        & (base_copy["EVT"] != "death")
    )
    durees_totales = base_copy.loc[conditions, "evt_duration"].to_numpy()
    if len(durees_totales):
        warn1 = "Careful ! Some EVT (other than in, out and death) appeared at the same TIMESTAMP for a same ID_PATIENT. "
        warn2 = (
            "For these co-occuring EVT, the duration of each of these 2 EVT will be half the "
            "duration separating them from the next EVT of this patient. "
        )
        warn3 = (
            "For example, if a patient has T1=A, T1=B, T6=C, then the duration will be "
            "A: 2T, B: 3T, and the timeline from D1 will be A-A-B-B-B-C"
        )
        logger.warning(warn1 + warn2 + warn3)
    durees_divisees_par_deux = np.ceil(
        base_copy.loc[conditions, "evt_duration"].to_numpy() / 2
    )
    complementaire_des_durees = [
        tot - div_par_deux
        for tot, div_par_deux in zip(
            durees_totales, durees_divisees_par_deux, strict=False
        )
    ]
    base_copy.loc[conditions, "evt_duration"] = durees_divisees_par_deux
    conditions2 = (
        (base_copy["EVT"] != "in")
        & (base_copy["EVT"] != "start")
        & base_copy["evt_duration"].eq(0)
        & (base_copy["EVT"].shift(-1) != "out")
        & (base_copy["EVT"].shift(-1) != "end")
        & (base_copy["EVT"].shift(-1) != "death")
    )
    try:
        base_copy.loc[conditions2, "evt_duration"] = complementaire_des_durees
    except ValueError as exc:
        pat = base_copy.loc[conditions2, "ID_PATIENT"].unique()
        raise ValueError(
            f"Treatment durations are inconsistent for patients {pat}"
        ) from exc

    base_copy.loc[
        base_copy["ID_PATIENT"].ne(base_copy["ID_PATIENT"].shift(-1)),
        "evt_duration",
    ] = np.nan

    return base_copy


def _check_or_not(base: pd.DataFrame, check=True) -> pd.DataFrame:
    """Validate the dataset and check it if ``check=True``; otherwise only reorder.

    :param base: DataFrame — one row per event for a patient; columns:
        'ID_PATIENT', 'EVT', 'TIMESTAMP'
    :param check: set to False to skip validations (e.g., when there is neither
        'in' nor 'out')
    :return: reordered copy of ``base`` (by ID_PATIENT, TIMESTAMP)

    .. note:: Regardless of ``check`` being True or False, the dataset is reordered.
    """
    base_copy = (
        Checks(base).base
        if ("start" not in base["EVT"].unique() and check)
        else stable_sort(base)
    )

    return base_copy
