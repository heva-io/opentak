import numpy as np
import pandas as pd


class GenerateCohortTAK:
    def __init__(
        self, nb_patients=500, nb_days_end=365, random_state: int | None = None
    ):
        """Initialize the number of patients and the maximum length of follow-up.

        :param nb_patients: number of patients in the cohort
        :param nb_days_end: maximum number of days for which deliveries can be made
        """
        self.nb_patients = nb_patients
        self.nb_days_end = nb_days_end
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def initialisation_dataframe(self, treatment_name="A", dose_mean=30, dose_std=10):
        """Create an initial database with the correct columns and populate it with treatment_name entries.

        :param treatment_name: name of the treatment
        :param dose_mean: theoretical dosage, used as  average delivery interval
        :param dose_std: standard deviation of the interval between doses, represents the difference found in practice
        between the dosage and the interval between doses.
        """
        # Compute max deliveries
        nb_deliveries_max = int(self.nb_days_end / (dose_mean - dose_std / 2))

        # Compute number of days column based on dosage
        dose_real = self.rng.normal(
            dose_mean, dose_std, size=(self.nb_patients, nb_deliveries_max)
        ).astype(int)
        dose_real = np.where(dose_real <= 0, 1, dose_real)
        nbdays_not_flatten = np.cumsum(dose_real, axis=1)
        nbdays = nbdays_not_flatten.flatten()

        base = pd.DataFrame([])
        base["ID_PATIENT"] = np.repeat(range(self.nb_patients), nb_deliveries_max)
        base["TIMESTAMP"] = nbdays
        base["EVT"] = treatment_name
        base["DOSAGE"] = 30

        # remove nbdays later than nb_jours_max
        self.base = base[base["TIMESTAMP"].le(self.nb_days_end)]
        self.nb_rows_per_patient = (
            self.base.groupby("ID_PATIENT").count()["TIMESTAMP"].to_numpy()
        )

        return self.base

    def add_switch_linear(
        self,
        treatment_name,
        start_period_switch=None,
        end_period_switch=None,
        proportion_of_cohort=1,
    ):
        # ruff: noqa: D205
        """Add a switch to the treatment_name medication, with a linear distribution from start_period_switch to
           end_period_switch for the nbdays in which it appears.

        :param treatment_name: name of the treatment
        :param start_period_switch: smallest nbdays of switch
        :param end_period_switch: largest nbdays os switch
        :param proportion_of_cohort: proportion of the cohort affected by the switch
        """
        # Calculation of a start_period_switch and an end_period_switch if they are not provided by the user
        if start_period_switch is None:
            start_period_switch = int(self.nb_days_end / 3)
        if end_period_switch is None:
            end_period_switch = int(2 * self.nb_days_end / 3)

        # calculation of the distribution of change days to treatment_name
        distrib_nbdays_switch = self.rng.integers(
            start_period_switch, end_period_switch, size=self.nb_patients
        )

        # Applies the switch to proportion_of_cohort and adds the lines to the dataframe.
        return self._add_switch(
            treatment_name, distrib_nbdays_switch, proportion_of_cohort
        )

    def add_switch_gaussien(
        self, treatment_name, mean=None, std=None, proportion_of_cohort=1
    ):
        # ruff: noqa: D205
        """Add a switch to the treatment_name medication, with a Gaussian distribution for the nbdays
        at which it appears.

        :param treatment_name: name of the treatment
        :param mean: mean of the Gaussian distribution of switch nbdays
        :param std: standard deviation of the Gaussian distribution of switch nbdays
        :param proportion_of_cohort: proportion of the cohort affected by the switch
        """
        # Calculating a mean and standard deviation if they are not provided by the user
        if mean is None:
            mean = int(self.nb_days_end / 2)
        if std is None:
            std = int(self.nb_days_end / 8)

        # Calculation of the distribution of change days to treatment_name
        distrib_nbdays_switch = self.rng.normal(mean, std, size=self.nb_patients)

        # Applies the switch to proportion_of_cohort and adds the lines to the dataframe.
        return self._add_switch(
            treatment_name, distrib_nbdays_switch, proportion_of_cohort
        )

    def add_drug_holidays(
        self,
        start_dh_min: int | None = None,
        start_dh_max: int | None = None,
        duration_dh_min: int | None = None,
        duration_dh_max: int | None = None,
        proportion_of_cohort=1,
    ):
        """Remove deliveries in order to show drug holidays.

        :param start_dh_min: minimum number of days from the start of the drug holiday period
        :param start_dh_max: maximum number of days from the start of the drug holiday period
        :param duration_dh_min: minimum duration of the drug holiday period
        :param duration_dh_max: maximum duration of the drug holiday period
        :param proportion_of_cohort: proportion of the cohort affected by the drug holidays
        """
        # Calculation of start_dh_min and start_dh_max if not provided by the user
        if start_dh_min is None:
            start_dh_min = int(self.nb_days_end / 3)
        if start_dh_max is None:
            start_dh_max = int(2 * self.nb_days_end / 3)

        # Calculation of a start_period_switch and an end_period_switch if they are not provided by the user
        if duration_dh_min is None:
            duration_dh_min = int(self.nb_days_end / 6)
        if duration_dh_max is None:
            duration_dh_max = int(1.2 * self.nb_days_end / 6)

        # Calculation of the distribution of treatment interruption days
        distrib_nbdays_start_dh = self.rng.integers(
            start_dh_min, start_dh_max, size=self.nb_patients
        )
        duration_dh = self.rng.integers(
            duration_dh_min, duration_dh_max, size=self.nb_patients
        )
        distrib_nbdays_end_dh = distrib_nbdays_start_dh + duration_dh

        # check that proportion_of_cohort belongs to the segment [0,1]
        if proportion_of_cohort < 0 or proportion_of_cohort > 1:
            raise AttributeError("proportion_of_cohort should be between 0 and 1")

        # Random selection of patients who will not receive this switch
        index_droped = self.rng.choice(
            self.nb_patients,
            int((1 - proportion_of_cohort) * self.nb_patients),
            replace=False,
        )
        distrib_nbdays_end_dh[index_droped] = distrib_nbdays_start_dh[index_droped]

        # Add the treatment_name lines corresponding to the switch in the database.
        self.base["nbdays_start_dh"] = np.repeat(
            distrib_nbdays_start_dh, self.nb_rows_per_patient
        )
        self.base["nbdays_end_dh"] = np.repeat(
            distrib_nbdays_end_dh, self.nb_rows_per_patient
        )
        self.base = self.base[
            self.base["TIMESTAMP"].le(self.base["nbdays_start_dh"])
            | self.base["TIMESTAMP"].ge(self.base["nbdays_end_dh"])
        ]
        self.base = self.base.drop(["nbdays_start_dh", "nbdays_end_dh"], axis=1)

        self._update_nb_rows_per_patient()

        return self.base

    def _update_nb_rows_per_patient(self):
        self.nb_rows_per_patient = (
            self.base.groupby("ID_PATIENT").count()["TIMESTAMP"].to_numpy()
        )

    def _add_switch(self, treatment_name, distrib_nbdays_switch, proportion_of_cohort):
        """Add a switch to the treatment_name drug.

        :param treatment_name: name of the treatment
        :param distrib_nbdays_switch: distribution of nbdays on which the switch occurs, in order of ID_PATIENT
        :param proportion_of_cohort: proportion of the cohort affected by the switch
        """
        # check that proportion_of_cohort belongs to the segment [0,1]
        if proportion_of_cohort < 0 or proportion_of_cohort > 1:
            raise AttributeError("proportion_of_cohort should be between 0 and 1")

        # Random selection of patients who will not receive this switch
        index_droped = self.rng.choice(
            self.nb_patients,
            int((1 - proportion_of_cohort) * self.nb_patients),
            replace=False,
        )
        distrib_nbdays_switch[index_droped] = self.nb_days_end + 1

        # Addition of treatment_name lines corresponding to the switch in the database
        self.base["switch"] = np.repeat(distrib_nbdays_switch, self.nb_rows_per_patient)
        self.base.loc[self.base["switch"].le(self.base["TIMESTAMP"]), "EVT"] = (
            treatment_name
        )
        self.base = self.base.drop("switch", axis=1)
        return self.base

    def drop_missing_deliveries(self, proba_suppression_delivery=0.05):
        """Randomly deletes entries from the database.

        :param proba_suppression_delivery: proportion of deliveries to be removed from the database
        """
        # Remove deliveries at random
        self.base = self.base.loc[
            self.rng.random(len(self.base)) > proba_suppression_delivery, :
        ]

        self._update_nb_rows_per_patient()

        return self.base

    def add_in_out(self, proba_death=0):
        """Add a “in” to nbdays = 0 at the start of each patient and an “out” or “death” to nb_days_end+1 for each
        patient.

        :param proba_death: proportion of deaths in the cohort
        """
        # add 'in'
        base_in = pd.DataFrame(list(range(self.nb_patients)), columns=["ID_PATIENT"])
        base_in["TIMESTAMP"] = 0
        base_in["EVT"] = "in"
        base_in["DOSE"] = 0

        # add out and death
        base_out = pd.DataFrame(list(range(self.nb_patients)), columns=["ID_PATIENT"])
        base_out["TIMESTAMP"] = self.rng.integers(
            self.nb_days_end + 1, size=len(base_out)
        )
        base_out["EVT"] = self.rng.choice(
            ["out", "death"], self.nb_patients, p=[1 - proba_death, proba_death]
        )
        base_out.loc[base_out["EVT"].eq("out"), "TIMESTAMP"] = self.nb_days_end + 1
        base_out["DOSE"] = np.nan

        # remove element appearing after death or out
        self.base["end"] = np.repeat(
            base_out["TIMESTAMP"].values, self.nb_rows_per_patient
        )
        self.base = self.base[self.base["TIMESTAMP"].le(self.base["end"])]
        self.base = self.base.drop("end", axis=1)

        self.base = pd.concat([base_in, self.base, base_out]).sort_values(
            ["ID_PATIENT", "TIMESTAMP"], kind="mergesort"
        )

        self._update_nb_rows_per_patient()

        return self.base
