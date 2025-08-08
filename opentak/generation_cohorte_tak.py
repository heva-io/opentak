import numpy as np
import pandas as pd


class GenerationCohorteTAK:
    def __init__(
        self, nb_patients=500, nb_jours_end=365, random_state: int | None = None
    ):
        """Initialise le nombre de patients et la longueur max du suivi.

        :param nb_patients: nombre de patients dans la cohorte
        :param nb_jours_end: nbjours maximal auquel on peut avoir des délivrances
        :param random_state: random seed for reproducible results
        """
        self.nb_patients = nb_patients
        self.nb_jours_end = nb_jours_end
        self.random_state = random_state
        self.rng = np.random.default_rng(self.random_state)

    def initialisation_dataframe(self, nom_traitement="A", pos_moyenne=30, pos_std=10):
        """Crée une base initiale avec les bonnes colonnes et remplies de délivrances de nom_traitement.

        :param nom_traitement: nom du médicament
        :param pos_moyenne: posologie théorique, utilisée comme durée inter délivrance moyenne
        :param pos_std: standart deviation de la durée interdélivrance, représente l'écart que l'on trouve dans la
        pratique entre le posologie et la durée interdelivrance.
        :return: DataFrame with initial treatment data
        """
        # Calcul du delivrance max
        nb_delivrances_max = int(self.nb_jours_end / (pos_moyenne - pos_std / 2))

        # Calcul colonne NBJOURS à base de posologie
        pos_reelles = self.rng.normal(
            pos_moyenne, pos_std, size=(self.nb_patients, nb_delivrances_max)
        ).astype(int)
        pos_reelles = np.where(pos_reelles <= 0, 1, pos_reelles)
        nbjours_not_flatten = np.cumsum(pos_reelles, axis=1)
        nbjours = nbjours_not_flatten.flatten()

        # Mise en place de la dataframe
        base = pd.DataFrame([])
        base["ID_PATIENT"] = np.repeat(range(self.nb_patients), nb_delivrances_max)
        base["TIMESTAMP"] = nbjours
        base["EVT"] = nom_traitement
        base["POSOLOGIE"] = 30

        # enlever les nbjours plus tardifs que nb_jours_max
        self.base = base[base["TIMESTAMP"].le(self.nb_jours_end)]
        self.nb_rows_per_patient = (
            self.base.groupby("ID_PATIENT").count()["TIMESTAMP"].to_numpy()
        )

        # renvoie la base
        return self.base

    def add_switch_linear(
        self,
        nom_traitement,
        start_period_switch=None,
        end_period_switch=None,
        proportion_of_cohort=1,
    ):
        """Add a switch to the drug nom_traitement, with a linear distribution.

        :param nom_traitement: name of the drug
        :param start_period_switch: earliest day of switch
        :param end_period_switch: latest day of switch
        :param proportion_of_cohort: proportion of the cohort affected by the switch
        :return: Updated DataFrame with switch events
        """
        # Calcul d'une start_period_switch et d'un end_period_switch s'ils ne sont pas donnés par l'utilisateur
        if start_period_switch is None:
            start_period_switch = int(self.nb_jours_end / 3)
        if end_period_switch is None:
            end_period_switch = int(2 * self.nb_jours_end / 3)

        # calcul de la distribution du jours de changement vers nom_traitement
        distrib_nbjours_switch = self.rng.integers(
            start_period_switch, end_period_switch, size=self.nb_patients
        )

        # Applique le switch a proportion_of_cohort et ajoute les lignes dans le dataframe
        return self._add_switch(
            nom_traitement, distrib_nbjours_switch, proportion_of_cohort
        )

    def add_switch_gaussien(
        self, nom_traitement, mean=None, std=None, proportion_of_cohort=1
    ):
        """Add a switch to the drug nom_traitement, with a Gaussian distribution for the days on which it appears.

        :param nom_traitement: name of the drug
        :param mean: mean of the Gaussian distribution for the switch days
        :param std: standard deviation of the Gaussian distribution for the switch days
        :param proportion_of_cohort: proportion of the cohort affected by the switch
        :return: Updated DataFrame with switch events
        """
        # Calcul d'une moyenne et d'un std s'ils ne sont pas donnés par l'utilisateur
        if mean is None:
            mean = int(self.nb_jours_end / 2)
        if std is None:
            std = int(self.nb_jours_end / 8)

        # calcul de la distribution du jours de changement vers nom_traitement
        distrib_nbjours_switch = self.rng.normal(mean, std, size=self.nb_patients)

        # Applique le switch a proportion_of_cohort et ajoute les lignes dans le dataframe
        return self._add_switch(
            nom_traitement, distrib_nbjours_switch, proportion_of_cohort
        )

    def add_drug_holidays(
        self,
        start_dh_min: int | None = None,
        start_dh_max: int | None = None,
        duration_dh_min: int | None = None,
        duration_dh_max: int | None = None,
        proportion_of_cohort=1,
    ):
        """Supprime des delivrances de manière à faire apparaitre des drugs holidays.

        :param start_dh_min: nbjours du début de la période de drugs holidays le plus petit
        :param start_dh_max: nbjours du début de la période de drugs holidays le plus grand
        :param duration_dh_min: durée min de la période de drug holidays
        :param duration_dh_max: durée max de la période de drug holidays
        :param proportion_of_cohort: Proportion de la cohorte concernée par le drug holidays
        :return: Updated DataFrame with drug holiday periods
        """
        # Calcul d'une start_dh_min et d'un start_dh_max s'ils ne sont pas donnés par l'utilisateur
        if start_dh_min is None:
            start_dh_min = int(self.nb_jours_end / 3)
        if start_dh_max is None:
            start_dh_max = int(2 * self.nb_jours_end / 3)

        # Calcul d'une start_period_switch et d'un end_period_switch s'ils ne sont pas donnés par l'utilisateur
        if duration_dh_min is None:
            duration_dh_min = int(self.nb_jours_end / 6)
        if duration_dh_max is None:
            duration_dh_max = int(1.2 * self.nb_jours_end / 6)

        # calcul de la distribution du jours d'arret de traitement
        distrib_nbjours_start_dh = self.rng.integers(
            start_dh_min, start_dh_max, size=self.nb_patients
        )
        duration_dh = self.rng.integers(
            duration_dh_min, duration_dh_max, size=self.nb_patients
        )
        distrib_nbjours_end_dh = distrib_nbjours_start_dh + duration_dh

        # checks que proportion_of_cohort appartient bien au segment [0,1]
        if proportion_of_cohort < 0 or proportion_of_cohort > 1:
            raise AttributeError("proportion_of_cohort should be between 0 and 1")

        # Tirage au sort des patients qui n'auront pas ce switch
        index_droped = self.rng.choice(
            self.nb_patients,
            int((1 - proportion_of_cohort) * self.nb_patients),
            replace=False,
        )
        distrib_nbjours_end_dh[index_droped] = distrib_nbjours_start_dh[index_droped]

        # Ajout des lignes de nom_traitement correpondant au switch dans la base
        self.base["nbjours_start_dh"] = np.repeat(
            distrib_nbjours_start_dh, self.nb_rows_per_patient
        )
        self.base["nbjours_end_dh"] = np.repeat(
            distrib_nbjours_end_dh, self.nb_rows_per_patient
        )
        self.base = self.base[
            self.base["TIMESTAMP"].le(self.base["nbjours_start_dh"])
            | self.base["TIMESTAMP"].ge(self.base["nbjours_end_dh"])
        ]
        self.base = self.base.drop(["nbjours_start_dh", "nbjours_end_dh"], axis=1)

        # actualisation du nombre de lignes par patient
        self._actualiser_nb_rows_per_patient()

        return self.base

    def _actualiser_nb_rows_per_patient(self):
        self.nb_rows_per_patient = (
            self.base.groupby("ID_PATIENT").count()["TIMESTAMP"].to_numpy()
        )

    def _add_switch(self, nom_traitement, distrib_nbjours_switch, proportion_of_cohort):
        """Ajoute un switch vers le medicament nom_traitement.

        :param nom_traitement: nom du médicament
        :param distrib_nbjours_switch: distribution des nbjours auxquels le switch a lieu, dans l'ordre des ID_PATIENT
        :param proportion_of_cohort: Proportion de la cohorte concernée par le switch
        """
        # checks que proportion_of_cohort appartient bien au segment [0,1]
        if proportion_of_cohort < 0 or proportion_of_cohort > 1:
            raise AttributeError("proportion_of_cohort should be between 0 and 1")

        # Tirage au sort des patients qui n'auront pas ce switch
        index_droped = self.rng.choice(
            self.nb_patients,
            int((1 - proportion_of_cohort) * self.nb_patients),
            replace=False,
        )
        distrib_nbjours_switch[index_droped] = self.nb_jours_end + 1

        # Ajout des lignes de nom_traitement coorepondant au switch dans la base
        self.base["switch"] = np.repeat(
            distrib_nbjours_switch, self.nb_rows_per_patient
        )
        self.base.loc[self.base["switch"].le(self.base["TIMESTAMP"]), "EVT"] = (
            nom_traitement
        )
        self.base = self.base.drop("switch", axis=1)
        return self.base

    def drop_missing_deliveries(self, proba_suppression_delivery=0.05):
        """Supprime des délivrances dans la base, de manière aléatoire.

        :param proba_suppression_delivery: proportion de délivrances à supprimer de la base
        """
        # Enlève des délivrances de manière aléatoire
        self.base = self.base.loc[
            self.rng.random(len(self.base)) > proba_suppression_delivery, :
        ]

        # actualisation du nombre de lignes par patient
        self._actualiser_nb_rows_per_patient()

        return self.base

    def add_in_out(self, proba_death=0):
        """Add in and out events to the cohort.

        Adds an 'in' event at TIMESTAMP = 0 for each patient and an 'out' or 'death'
        event at nb_jours_end+1 for each patient.

        :param proba_death: proportion of deaths in the cohort
        """
        # ajout 'in'
        base_in = pd.DataFrame(list(range(self.nb_patients)), columns=["ID_PATIENT"])
        base_in["TIMESTAMP"] = 0
        base_in["EVT"] = "in"
        base_in["POSOLOGIE"] = 0

        # ajout out et death
        # TODO : mettre une variabilité possible sur le out
        base_out = pd.DataFrame(list(range(self.nb_patients)), columns=["ID_PATIENT"])
        base_out["TIMESTAMP"] = self.rng.integers(
            self.nb_jours_end + 1, size=len(base_out)
        )
        base_out["EVT"] = self.rng.choice(
            ["out", "death"], self.nb_patients, p=[1 - proba_death, proba_death]
        )
        base_out.loc[base_out["EVT"].eq("out"), "TIMESTAMP"] = self.nb_jours_end + 1
        base_out["POSOLOGIE"] = np.nan

        # enlever element apparaissant après death ou out
        self.base["end"] = np.repeat(
            base_out["TIMESTAMP"].values, self.nb_rows_per_patient
        )
        self.base = self.base[self.base["TIMESTAMP"].le(self.base["end"])]
        self.base = self.base.drop("end", axis=1)

        # concatenation
        self.base = pd.concat([base_in, self.base, base_out]).sort_values(
            ["ID_PATIENT", "TIMESTAMP"], kind="mergesort"
        )

        # actualisation du nombre de lignes par patient
        self._actualiser_nb_rows_per_patient()

        return self.base
