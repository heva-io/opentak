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
        """Initialise la classe faisant les checks sur la base initiale.

        Attention, il ordonne la base selon ID_PATIENT et TIMESTAMP, en laissant in devant et out, death derriere.

        :param base: (df) une ligne = un evenement pour un patient, colonnes : 'ID_PATIENT', 'EVT' et 'TIMESTAMP'
        :param check_unique_in_for_everybody: run check method check_in_for_everybody
        :param check_unique_out_or_death_for_everybody: run check method check_out_for_everybody
        :param reorder: reorder base according to ordonne method
        :param check_no_in_after_treatment: run check method check_no_in_after_treatment
        (only valid if check_unique_in_for_everybody = True and reorder = True)
        :param check_no_out_before_treatment: run check method check_no_out_before_treatment
        (only valid if check_unique_out_or_death_for_everybody = True and reorder = True)
        :param check_no_duplicated_rows: run check method check_no_duplicated_rows
        :param check_mutiple_delivrance_on_same_date: run check method check_mutiple_delivrance_on_same_date
        (only valid if reorder = True)
        :return: None (raise Error is checks invalid)
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
                    "If check_no_in_after_treatment = True, \
                    reorder and check_unique_in_for_everybody should be set to True"
                )
            self.check_no_in_after_treatment()

        if check_no_out_before_treatment:
            if not check_unique_out_or_death_for_everybody or not reorder:
                raise ValueError(
                    "If check_no_out_before_treatment = True, \
                    reorder and check_unique_out_or_death_for_everybody should be set to True"
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
        """Check si tous les patients ont bien un 'in' et un unique.

        :raise ValueError: si un patient un plusieur
        """
        # Recuperer liste des patients, et liste des patients ayant un 'in'
        set_patients = set(self.base["ID_PATIENT"])
        list_patients_in = list(
            self.base[self.base["EVT"].eq("in")]["ID_PATIENT"].values
        )

        # Cas où il y a plus de in que de patients (ou alors des patients avec plusieurs in),
        # ET/OU des patients sans in
        patient_sans_in = set_patients - set(list_patients_in)
        if len(patient_sans_in):
            raise ValueError(
                f"Attention : les patients {patient_sans_in} n'ont pas de 'in'"
            )

        pat_plusieurs_in = {
            pat for pat, nb_in in Counter(list_patients_in).items() if nb_in != 1
        }
        if pat_plusieurs_in:
            raise ValueError(
                f"Attention : les patients {pat_plusieurs_in} ont plusieurs 'in' chacun"
            )

    def check_out_for_everybody(
        self,
    ) -> None:
        """Check si tous les patients ont bien un 'out' ou un 'death' (et un unique)."""
        # Recuperer liste des patients, et liste des patients ayant un 'out' et liste des patients ayant un 'death'
        set_patients = set(self.base["ID_PATIENT"])
        list_patients_out_by_out = list(
            self.base[self.base["EVT"].eq("out")]["ID_PATIENT"].values
        )
        list_patients_out_by_death = list(
            self.base[self.base["EVT"].eq("death")]["ID_PATIENT"].values
        )
        list_patients_out = list_patients_out_by_out + list_patients_out_by_death

        # Cas où il y a plus de out/death que de patients
        # (ou alors des patients avec plusieurs out/death), ET/OU des patients sans out/death
        patient_sans_out = set_patients - set(list_patients_out)
        if len(patient_sans_out):
            raise ValueError(
                f"Attention : les patients {patient_sans_out} n'ont pas de 'out'"
            )

        pat_plusieurs_out = {
            pat for pat, nb_in in Counter(list_patients_out).items() if nb_in != 1
        }
        if pat_plusieurs_out:
            raise ValueError(
                f"Attention : les patients {pat_plusieurs_out} ont plusieurs 'out' chacun"
            )

    def ordonne(
        self,
    ) -> pd.DataFrame:
        """Mettre les 'in' au debut et les 'out' et 'death' à la fin, puis trié en mergesort

        (algo stable) selon ID_PATIENT puis TIMESTAMP.
        """
        base_in = self.base[self.base["EVT"].eq("in")]
        base_pas_in_pas_out = self.base[~self.base["EVT"].isin(["in", "out", "death"])]
        base_out = self.base[self.base["EVT"].isin(["out", "death"])]
        return stable_sort(pd.concat([base_in, base_pas_in_pas_out, base_out]))

    def check_no_in_after_treatment(
        self,
    ) -> None:
        """Check si tous les patients sont bien 'in' avant leur premier EVT'.

        S'il y a des patients dont ce n'est pas le cas : donne leur ID et affiche
        les 4 premieres lignes les concernant + raise une ValueError

        :raise ValueError: s'il y a des in avant le dernier EVT
        """
        # repérer les patients ayant un 'in' apres leur premier traitement
        pat_in_after_treatment = self.base[
            self.base["ID_PATIENT"].eq(self.base["ID_PATIENT"].shift(+1))
            & self.base["EVT"].eq("in")
        ]["ID_PATIENT"].to_numpy()

        # S'il y en a : afficher leur premieres lignes, pour cibler le problème
        if len(pat_in_after_treatment):
            logger.error(
                "Les patients %s ont un 'in' apres leur premier traitement",
                pat_in_after_treatment,
            )

            for pat in pat_in_after_treatment:
                logger.error(self.base[self.base["ID_PATIENT"].eq(pat)].iloc[:4, :])

            raise ValueError(
                "Il y a des patients dont le 'in' est après le premier traitement"
            )

    def check_no_out_before_treatment(
        self,
    ) -> None:
        """Check si tous les patients sont bien 'out' après leur dernier traitement'.

        S'il y a des patients dont ce n'est pas le cas : donne leur ID et
        affiche les 4 dernières lignes les concernant + raise ValueError

        :raise ValueError: s'il y a des out apres le dernier traitement
        """
        # repérer les patients ayant un 'out' avant leur dernier traitement
        pat_out_before_treatment = self.base[
            self.base["ID_PATIENT"].eq(self.base["ID_PATIENT"].shift(-1))
            & self.base["EVT"].eq("out")
        ]["ID_PATIENT"].to_numpy()

        # S'il y en a : afficher leur dernières lignes, pour cibler le problème
        if len(pat_out_before_treatment):
            logger.error(
                "Les patients %s ont un 'out' avant leur dernier traitement",
                pat_out_before_treatment,
            )
            for pat in pat_out_before_treatment:
                logger.error(self.base[self.base["ID_PATIENT"].eq(pat)].iloc[-4:, :])

            raise ValueError(
                "Il y a des patients dont le 'out' est avant le dernier traitement"
            )

    def check_no_duplicated_rows(
        self,
    ) -> None:
        """Check si la base n'a pas de lignes dupliquées."""
        try:
            assert not self.base.duplicated().any()  # noqa: S101
        except AssertionError as exc:
            pat_duplicated = self.base[self.base.duplicated()]["ID_PATIENT"].unique()
            raise ValueError(
                f"Il y a des lignes en doubles dans le dataframe (id patients : {pat_duplicated})"
            ) from exc

    def check_mutiple_delivrance_on_same_date(
        self,
    ) -> bool:
        """Check si la base contient des patients ayant un même jour plusieurs

        délivrances de médicament différents.


        :return Booleen: True s'il y a des multiples délivrances, False sinon
        """
        base_cop = self.base.copy()

        # Capter les patients ayant 2 délivrances le même jour
        conditions = (
            base_cop["ID_PATIENT"].eq(base_cop["ID_PATIENT"].shift(-1))
            & base_cop["TIMESTAMP"].eq(base_cop["TIMESTAMP"].shift(-1))
            & (base_cop["EVT"].shift(-1) != "out")
            & (base_cop["EVT"] != "in")
        )

        if len(base_cop[conditions]):
            patients_concernes = base_cop[conditions]["ID_PATIENT"].unique()
            logger.info(
                "Il y a %s délivrances de 2 médicaments différents au même jour (in et out exclus)",
                len(base_cop[conditions]),
            )
            logger.debug("Ce sont les patients %s", patients_concernes)

            # voir 3 délivrances ou plus le même jour (seulement si au moins deux ont déjà été captées)
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
                    "Les patients %s ont 3 délivrances le même jour ou plus \
                    (in et out exclus)",
                    patients_concernes_triple,
                )
                raise ValueError(
                    "Il y a %s délivrances de 3 médicaments différents (ou plus) \
                    au même jour (in et out exclus)",
                    len(base_cop[conditions_triple]),
                )
            return True
        return False


def _check_or_not(base: pd.DataFrame, check=True) -> pd.DataFrame:
    """Check base (presence de in, out, death, etc) et l'ordonne si check=True, sinon ne fait que l'ordonner.

    :param base: (df) une ligne = un évènement pour un patient, colonnes : 'ID_PATIENT', 'EVT' et 'TIMESTAMP'
    :param check: mettre à False si on veut ne pas passer par l'étape de check
    (par exemple si nous n'avons ni in ni out).
    :return: (df) copie de base réordonner (ID_PATIENT, TIMESTAMP)

    ..note: Attention, que check soit a True ou à False, base est réordonnée
    """
    base_copy = (
        Checks(base).base
        if ("start" not in base["EVT"].unique() and check)
        else stable_sort(base)
    )

    return base_copy
