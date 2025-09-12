from collections import Counter

import pandas as pd

from opentak.logger import logger
from opentak.utils_events.preprocessing import stable_sort


class Checks:
    def __init__(
        self,
        base,
        check_unique_in_for_everybody: bool = True,
        check_unique_out_or_death_for_everybody: bool = True,
        reorder: bool = True,
        check_no_in_after_treatment: bool = True,
        check_no_out_before_treatment: bool = True,
        check_no_duplicated_rows: bool = True,
        check_mutiple_delivrance_on_same_date: bool = True,
    ) -> None:
        """Initialize the class that runs checks on the initial dataset.

        Note: It orders the dataset by ID_PATIENT then TIMESTAMP, placing 'in' events first and 'out'/'death' events last.

        :param base: DataFrame — one row per event for a patient; columns: 'ID_PATIENT', 'EVT', 'TIMESTAMP'
        :param check_unique_in_for_everybody: run check method ``check_in_for_everybody``
        :param check_unique_out_or_death_for_everybody: run check method ``check_out_for_everybody``
        :param reorder: reorder dataset according to ``ordonne`` method
        :param check_no_in_after_treatment: run check method ``check_no_in_after_treatment``
            (only valid if ``check_unique_in_for_everybody=True`` and ``reorder=True``)
        :param check_no_out_before_treatment: run check method ``check_no_out_before_treatment``
            (only valid if ``check_unique_out_or_death_for_everybody=True`` and ``reorder=True``)
        :param check_no_duplicated_rows: run check method ``check_no_duplicated_rows``
        :param check_mutiple_delivrance_on_same_date: run check method ``check_mutiple_delivrance_on_same_date``
            (only valid if ``reorder=True``)
        :return: None (raises an error if any check fails)
        """
        self.base = base

        if check_unique_in_for_everybody:
            self.check_in_for_everybody()
        if check_unique_out_or_death_for_everybody:
            self.check_out_for_everybody()

        if reorder:
            self.base = self.ordonne()

        if check_no_in_after_treatment:
            if not check_unique_in_for_everybody or not reorder:
                raise ValueError(
                    "If check_no_in_after_treatment = True, reorder and check_unique_in_for_everybody should be set to True"
                )
            self.check_no_in_after_treatment()

        if check_no_out_before_treatment:
            if not check_unique_out_or_death_for_everybody or not reorder:
                raise ValueError(
                    "If check_no_out_before_treatment = True, reorder and check_unique_out_or_death_for_everybody should be set to True"
                )
            self.check_no_out_before_treatment()

        if check_no_duplicated_rows:
            self.check_no_duplicated_rows()

        if check_mutiple_delivrance_on_same_date:
            if not reorder:
                raise ValueError(
                    "If check_mutiple_delivrance_on_same_date = True, reorder should be set to True"
                )
            self.check_mutiple_delivrance_on_same_date()

    def check_in_for_everybody(
        self,
    ) -> None:
        """Verify that every patient has exactly one 'in' event.

        :raises ValueError: if a patient is missing an 'in' or has more than one
        """
        set_patients = set(self.base["ID_PATIENT"])
        list_patients_in = list(
            self.base[self.base["EVT"].eq("in")]["ID_PATIENT"].values
        )

        patient_without_in = set_patients - set(list_patients_in)
        if len(patient_without_in):
            raise ValueError(
                f"Attention : Patients {patient_without_in} do not have 'in'"
            )

        pat_several_in = {
            pat for pat, nb_in in Counter(list_patients_in).items() if nb_in != 1
        }
        if pat_several_in:
            raise ValueError(
                f"Attention : Patients {pat_several_in} have multiple 'in'"
            )

    def check_out_for_everybody(
        self,
    ) -> None:
        """Verify that every patient has exactly one 'out' or one 'death' event.

        :raises ValueError: if a patient is missing an 'out' or 'death' event or has more than one
        """
        set_patients = set(self.base["ID_PATIENT"])
        list_patients_out_by_out = list(
            self.base[self.base["EVT"].eq("out")]["ID_PATIENT"].values
        )
        list_patients_out_by_death = list(
            self.base[self.base["EVT"].eq("death")]["ID_PATIENT"].values
        )
        list_patients_out = list_patients_out_by_out + list_patients_out_by_death

        patient_sans_out = set_patients - set(list_patients_out)
        if len(patient_sans_out):
            raise ValueError(
                f"Attention : Patients {patient_sans_out} do not have 'out'"
            )

        pat_plusieurs_out = {
            pat for pat, nb_in in Counter(list_patients_out).items() if nb_in != 1
        }
        if pat_plusieurs_out:
            raise ValueError(
                f"Attention : Patients {pat_plusieurs_out} have multiple 'out'"
            )

    def ordonne(
        self,
    ) -> pd.DataFrame:
        """Place 'in' events first and 'out'/'death' events last, then stably sort by ID_PATIENT then TIMESTAMP (mergesort).

        :return: Sorted eventlog
        """
        base_in = self.base[self.base["EVT"].eq("in")]
        base_pas_in_pas_out = self.base[~self.base["EVT"].isin(["in", "out", "death"])]
        base_out = self.base[self.base["EVT"].isin(["out", "death"])]
        return stable_sort(pd.concat([base_in, base_pas_in_pas_out, base_out]))

    def check_no_in_after_treatment(
        self,
    ) -> None:
        """Ensure each patient's 'in' occurs before their first treatment event.

        If violations are found: logs the patient IDs and the first four related rows, then raises ValueError.

        :raises ValueError: if an 'in' appears after the first treatment
        """
        pat_in_after_treatment = self.base[
            self.base["ID_PATIENT"].eq(self.base["ID_PATIENT"].shift(+1))
            & self.base["EVT"].eq("in")
        ]["ID_PATIENT"].to_numpy()

        if len(pat_in_after_treatment):
            logger.error(
                "Patients %s have an 'in' after their first treatment",
                pat_in_after_treatment,
            )

            for pat in pat_in_after_treatment:
                logger.error(self.base[self.base["ID_PATIENT"].eq(pat)].iloc[:4, :])

            raise ValueError("Some patients have 'in' after their first treatment")

    def check_no_out_before_treatment(
        self,
    ) -> None:
        """Ensure each patient's 'out' occurs after their last treatment event.

        If violations are found: logs the patient IDs and the last four related rows, then raises ValueError.

        :raises ValueError: if an 'out' appears before the last treatment
        """
        pat_out_before_treatment = self.base[
            self.base["ID_PATIENT"].eq(self.base["ID_PATIENT"].shift(-1))
            & self.base["EVT"].eq("out")
        ]["ID_PATIENT"].to_numpy()

        if len(pat_out_before_treatment):
            logger.error(
                "Patients %s have an 'out' before their last treatment",
                pat_out_before_treatment,
            )
            for pat in pat_out_before_treatment:
                logger.error(self.base[self.base["ID_PATIENT"].eq(pat)].iloc[-4:, :])

            raise ValueError("Some patients have 'out' before their last treatment")

    def check_no_duplicated_rows(
        self,
    ) -> None:
        """Ensure the dataset has no duplicated rows.

        :raises ValueError: if duplicate rows are found
        """
        try:
            assert not self.base.duplicated().any()  # noqa: S101
        except AssertionError as exc:
            pat_duplicated = self.base[self.base.duplicated()]["ID_PATIENT"].unique()
            raise ValueError(
                f"There are duplicate rows in the DataFrame (patient IDs: {pat_duplicated})"
            ) from exc

    def check_mutiple_delivrance_on_same_date(
        self,
    ) -> bool:
        """Check whether any patients have multiple deliveries of different medications on the same day.

        :return: bool — True if multiple deliveries exist, False otherwise
        :raises ValueError: if three or more deliveries occur the same day (excluding 'in' and 'out')
        """
        base_cop = self.base.copy()

        conditions = (
            base_cop["ID_PATIENT"].eq(base_cop["ID_PATIENT"].shift(-1))
            & base_cop["TIMESTAMP"].eq(base_cop["TIMESTAMP"].shift(-1))
            & (base_cop["EVT"].shift(-1) != "out")
            & (base_cop["EVT"] != "in")
        )

        if len(base_cop[conditions]):
            patients_concernes = base_cop[conditions]["ID_PATIENT"].unique()
            logger.info(
                "There are %s deliveries of 2 different medications on the same day (excluding 'in' and 'out')",
                len(base_cop[conditions]),
            )
            logger.debug("It concerns the following patients %s", patients_concernes)

            conditions_triple = (
                base_cop["ID_PATIENT"].eq(base_cop["ID_PATIENT"].shift(-2))
                & base_cop["TIMESTAMP"].eq(base_cop["TIMESTAMP"].shift(-2))
                & (base_cop["EVT"].shift(-2) != "out")
                & (base_cop["EVT"] != "in")
            )
            if len(base_cop[conditions_triple]):
                patients_concernes_triple = base_cop[conditions_triple][
                    "ID_PATIENT"
                ].unique()
                logger.error(
                    "Patients %s have 3 or more deliveries on the same day (excluding 'in' and 'out')",
                    patients_concernes_triple,
                )
                raise ValueError(
                    "There are %s deliveries of 3 different medications (or more) on the same day (excluding 'in' and 'out')",
                    len(base_cop[conditions_triple]),
                )
            return True
        return False


def _check_or_not(base: pd.DataFrame, check=True) -> pd.DataFrame:
    """Validate the dataset (presence of 'in', 'out', 'death', etc.) and reorder it if ``check=True``; otherwise only reorder.

    :param base: DataFrame — one row per event for a patient; columns: 'ID_PATIENT', 'EVT', 'TIMESTAMP'
    :param check: set to False to skip validations (e.g., when there is neither 'in' nor 'out')
    :return: DataFrame — reordered copy (by ID_PATIENT, TIMESTAMP)

    .. note:: Regardless of ``check`` being True or False, the dataset is reordered.
    """
    base_copy = (
        Checks(base).base
        if ("start" not in base["EVT"].unique() and check)
        else stable_sort(base)
    )

    return base_copy
