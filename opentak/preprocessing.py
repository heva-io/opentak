from __future__ import annotations

from functools import wraps
from typing import Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from opentak.clustering import Tak, TakHca
from opentak.utils_events import checks
from opentak.utils_events.preprocessing import stable_sort
from opentak.utils_events.utils import add_evt_duration

pd.set_option("future.no_silent_downcasting", True)

_BASE_DICT = {
    "start": 0,
    "in": 1,
    "out": 2,
    "death": 3,
    "end": 4,
    "nothing": 5,
}


def reset_array(func):
    """Indicate that the output array should be recomputed regardless of the cache system.

    :param func: decorated preprocessing method
    :return: method with extra functionality
    """

    @wraps(func)
    def wrapper(*args):
        """Set the _must_create_array flag before calling the original method.

        :param args: arguments to pass to the decorated function
        :return: result of the decorated function
        """
        # ruff: noqa: SLF001
        args[0]._must_create_array = True
        return func(*args)

    return wrapper


def _validate_args(base: pd.DataFrame, max_days: int | None) -> None:
    """Validate arguments for TakBuilder initialization.

    :param base: event log dataframe to validate
    :param max_days: maximum days value to validate
    :raises ValueError: if required columns are missing from base or max_days is invalid
    """
    missing = {"TIMESTAMP", "ID_PATIENT", "EVT"} - set(base.columns)
    if missing:
        raise ValueError(f"'base' is missing the following columns: {missing}")

    if max_days is not None and max_days < 1:
        raise ValueError(f"'max_days' must be positive. Given '{max_days}'")


class TakBuilder:
    def __init__(
        self,
        base: pd.DataFrame,
        max_days: int | None = None,
    ) -> None:
        """Tak builder constructor

        :param base: event log as a dataframe
        :param max_days: patients' sequence max length in days
        """
        _validate_args(base, max_days)
        self.base = checks.Checks(base).base
        self.timescale = 1
        self.dict_label_id = dict(_BASE_DICT)
        self._create_dict_label_id()

        if not max_days:
            self.max_days = self.base.loc[self.base["EVT"].ne("end"), "TIMESTAMP"].max()
        else:
            self.max_days = max_days

        self._add_start()
        self._add_end()
        self.base = stable_sort(self.base)
        self.base = add_evt_duration(self.base)
        self.index_patients = self.base["ID_PATIENT"].unique()
        self.array: npt.NDArray | None = None
        self._must_create_array = True

    def build(
        self,
        kind: Literal["hca"] = "hca",
    ) -> Tak:
        """Build Tak object from builder.

        The ``kind`` parameter can take four values:

        ``"hca"``
            Classic Tak algorithm. Stands for Hierarchical Clustering Analysis.

        :param kind: kind always set to "hca" for Tak minimal release; other types will be introduced in next releases
        :return: Tak object
        """
        if self._must_create_array:
            self._create_array_from_evt_log()
            self._must_create_array = False

        kwargs = {
            "array": self.array,
            "index_patients": self.index_patients,
            "dict_label_id": self.dict_label_id,
            "timescale": self.timescale,
            "evt_log": self.base,
        }
        tak: Tak
        if kind == "hca":
            tak = TakHca(**kwargs)
        else:
            raise ValueError(
                "'kind' argument should be equal to 'hca' in this minimal release"
            )
        return tak

    def _create_dict_label_id(self):
        """Create the dict 'labels' -> 'id'.

        This method creates a mapping between event labels and their corresponding
        integer IDs, extending the base dictionary with new events found in the data.
        """
        sorted_evt = sorted(set(self.base["EVT"].unique()) - set(self.dict_label_id))

        events_ids = {
            evt: i for i, evt in enumerate(sorted_evt, start=len(self.dict_label_id))
        }

        self.dict_label_id.update(events_ids)

    def _add_start(self) -> None:
        """Add a 'start' event at 'TIMESTAMP' = 0 for all patients.

        We place this new event before the 'in' event to mark the beginning
        of each patient's timeline.
        """
        base_start = self.base.loc[
            self.base["EVT"].eq("in"), ["ID_PATIENT", "TIMESTAMP", "EVT"]
        ].copy()

        base_start["EVT"] = "start"
        base_start["TIMESTAMP"] = 0

        base_tak = pd.concat([base_start, self.base])
        self.base = base_tak

    def _add_end(self) -> None:
        """Add an 'end' event at 'TIMESTAMP' = max_days for all patients.

        This marks the end of the observation period for each patient's timeline.
        """
        base_end = self.base.loc[
            self.base["EVT"].isin(("out", "death")), ["ID_PATIENT", "TIMESTAMP", "EVT"]
        ].copy()

        base_end["EVT"] = "end"
        base_end["TIMESTAMP"] = self.max_days

        base_tak = pd.concat([self.base, base_end])
        self.base = base_tak

    def _create_array_from_evt_log(self) -> None:
        """Convert the event log to a dense matrix of treatments labels IDs for all patients.

        This method transforms the event log DataFrame into a numpy array where each row
        represents a patient and each column represents a timestamp. The array contains
        integer IDs corresponding to the events at each timestamp.

        :raises ValueError: if patient durations are invalid
        """
        list_patients: list[np.ndarray] = []
        for id_group, df_group in self.base.loc[self.base["EVT"] != "end"].groupby(
            "ID_PATIENT"
        ):
            evt = list(df_group["EVT"].replace(self.dict_label_id).astype("int"))
            durations = list(df_group["evt_duration"])

            if not all(
                isinstance(duration, (int, float)) for duration in durations
            ) or any(duration < 0 for duration in durations):
                raise ValueError(
                    f"Patient {id_group} has a non numeric or negative duration: {{df_group}}"
                )

            patient_sequence = np.repeat(evt, durations)

            list_patients.append(patient_sequence)

        self.array = np.array(list_patients, dtype=np.uint8)
