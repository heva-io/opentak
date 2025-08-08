import numpy as np
import pandas as pd

from tak.logger import logger
from tak.utils_events.checks import Checks
from tak.utils_events.preprocessing import stable_sort


def get_evt_log_with_frosenset_evt(
    log: pd.DataFrame,
    list_ignored_evt: list[str] | None = None,
) -> pd.Series:
    """Transform log pour assembler les evts concomittants et ne garder que ceux là

    (pour analyse).

    :param log: df avec comme colonnes "ID_PATIENT", "EVT" et "TIMESTAMP"
    :param list_ignored_evt: liste des evenements qui ne seront jamais considérés
    comme concomittant (par exemple 'in' ou 'out' ou 'death')
    :return: log avec uniquement les evenements concomittant
    (meme patient meme timestamp), et remplaçer avec leur frozenset
    """
    if list_ignored_evt is not None:
        log = log.query("EVT not in @list_ignored_evt")

    # Pour réduire temps de calcul :
    # ne travailler que sur les lignes qui nous interessent
    log_duplicated = log[log.duplicated(subset=["ID_PATIENT", "TIMESTAMP"], keep=False)]

    log_duplicated_grouped = log_duplicated.groupby(["ID_PATIENT", "TIMESTAMP"])["EVT"].agg(frozenset)

    return log_duplicated_grouped


def regroup_evt(
    log: pd.DataFrame,
    list_ignored_evt: list[str] | None = None,
    list_fzset_to_be_regrouped: list | None = None,
    sep: str = " + ",
) -> pd.DataFrame:
    """Assemble les evenements concommitant en un unique evenement "A + B".

    :param log: df avec comme colonnes "ID_PATIENT", "EVT" et "TIMESTAMP"
    :param list_ignored_evt: liste des evenements qui ne seront jamais considérés comme concomittant (par exemple 'in'
    ou 'out' ou 'death')
    :param list_fzset_to_be_regrouped: liste des elements à assembler ([frozenset({"A", "B"}), frozenset({"B", "D"})]).
    Si rien n'est donné, tous les evt concomittants sont assemblés.
    :param sep: séparateur quand les evts sont assemblés (sep = " + " -> "A" et "B" devient l'evenement "A + B")
    :return: log avec ces evts concomittants appartenant à list_fzset_to_be_regrouped assemblés en un unique evenement
    """
    log_cop = log.copy()

    frozen_set = pd.DataFrame(get_evt_log_with_frosenset_evt(log_cop, list_ignored_evt))

    if list_fzset_to_be_regrouped:
        # Prévenir si un element de list_ignored_evt est dans
        # list_fzset_to_be_regrouped: il ne sera pas pris en compte dans
        # list_fzset_to_be_regrouped
        if list_ignored_evt:
            elmt_of_list_fzset_to_be_regrouped = {x for fzset in list_fzset_to_be_regrouped for x in fzset}
            list_elmt_pb = elmt_of_list_fzset_to_be_regrouped & set(list_ignored_evt)
            if len(list_elmt_pb):
                logger.warning(
                    "Des élements de list_ignored_evt apparaissent dans list_fzset_to_be_regrouped, "
                    "ils seront ignorés (%s)",
                    list_elmt_pb,
                )

        # Ne garder que les evenements qui nous interessent
        frozen_set = frozen_set[frozen_set["EVT"].isin(list_fzset_to_be_regrouped)]

    if len(frozen_set):
        if list_ignored_evt is None:
            list_ignored_evt = []
        log_ignored = log_cop[log_cop["EVT"].isin(list_ignored_evt)].set_index(["ID_PATIENT", "TIMESTAMP"])
        log_not_ignored = log_cop[~log_cop["EVT"].isin(list_ignored_evt)].set_index(["ID_PATIENT", "TIMESTAMP"])

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
    """Reorder log so that according to ID_PATIENT and TIMESTAMP, with in first and out last when several EVT.

    :param log: df to be reordored
    :return: log reordored
    """
    log_in = log[log["EVT"].eq("in")]
    log_out = log[log["EVT"].eq("out")]
    log_other = log[~log["EVT"].isin(["in", "out"])].sort_values(["ID_PATIENT", "TIMESTAMP"])
    log_reorder = stable_sort(pd.concat([log_in, log_other, log_out]))

    return log_reorder


def deal_with_double_delivrance(base, reset_index=True) -> pd.DataFrame:
    """Inverse l'ordre des lignes d'un eventlog lorsqu'un patient a pour portion de séquence de traitement A-B-A

    avec le B-A donné au même TIMESTAMP. Idem pour A-B-A avec A-B donné au même TIMESTAMP.

    :param base: evtlog with ID_PATIENT, TIMESTAMP and EVT columns
    :param reset_index: reset_inde before returning the base (because rows could have be permutated)

    !!! note

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|y|B|
        |10|y|A|
        |...|...|...|

        avec x<y, devient

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|y|A|
        |10|y|B|
        |...|...|...|

        et

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|x|B|
        |10|y|A|
        |...|...|...|

        avec x<y, devient

        |ID_PATIENT|TIMESTAMP|EVT|
        | :--: | :--: | :--: |
        |...|...|...|
        |10|x|A|
        |10|x|A|
        |10|y|B|
        |...|...|...|

    :return: base dont l'ordre des lignes à été changé pour coller à la logique du traitement du patient.
    """
    base_cop = base.copy()
    try:
        pd.testing.assert_frame_equal(base_cop, base_cop.sort_values(["ID_PATIENT", "TIMESTAMP"], kind="mergesort"))
    except AssertionError:
        logger.warning(
            "The base was not sorted according to 'ID_PATIENT' and 'TIMESTAMP', it has been sorted \
                       To silent the warning, use base.sort_values(['ID_PATIENT', 'TIMESTAMP'], kind='mergesort')\
                        before this function"
        )

    # Gérer les cas des délivrances doubles où il parait plus intuitif d'en mettre une avant la deuxième
    conditions = (
        base_cop["ID_PATIENT"].eq(base_cop["ID_PATIENT"].shift(-1))
        & base_cop["TIMESTAMP"].eq(base_cop["TIMESTAMP"].shift(-1))
        & base_cop["EVT"].ne(base_cop["EVT"].shift(-1))
        & (base_cop["EVT"].eq(base_cop["EVT"].shift(-2)) | base_cop["EVT"].shift(+1).eq(base_cop["EVT"].shift(-1)))
        & ((base_cop["EVT"] != base_cop["EVT"].shift(+1)) & (base_cop["EVT"].shift(-1) != (base_cop["EVT"].shift(-2))))
    )

    # Switcher !
    conditions = conditions.reset_index(name="ttmt_to_move_down")
    conditions["new_index"] = conditions["index"]
    conditions["ttmt_to_move_up"] = conditions["ttmt_to_move_down"].shift(+1).fillna(False)
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
    """Rajoute une colonne avec le délai entre un évènement et le suivant (ici la durée de traitement).

    :param base: (df) une ligne = un évènement pour un patient, colonnes : 'ID_PATIENT', 'EVT' et 'TIMESTAMP'
    :param check: mettre à False si on veut ne pas passer par l'étape de check
    (par exemple si nous n'avons ni in ni out).
    :return: (df) copie de base avec la colonne evt_duration en plus

    ..note: Attention, que check soit a True ou à False, base est réordonnée car c'est indispensable
    """
    base_copy = _check_or_not(base, check)

    # calcul de la différence TIMESTAMP d'un evenement et NJOURS de celui en dessous de lui dans la dataframe
    base_copy["evt_duration"] = base_copy["TIMESTAMP"].shift(-1).diff().copy()

    # gerer le premier element du dataframe
    base_copy["evt_duration"].iloc[0] = base_copy["TIMESTAMP"].iloc[1] - base_copy["TIMESTAMP"].iloc[0]

    # gerer les doubles délivrances le même jour :
    # diviser la durée de traitement par le nombre de délivrance (2)
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
        warn1 = (
            "Careful ! Some EVT (other than in, out and death) appeared at the same TIMESTAMP for a same ID_PATIENT. "
        )
        warn2 = (
            "For these co-occuring EVT, the duration of each of these 2 EVT will be half the "
            "duration separating them from the next EVT of this patient. "
        )
        warn3 = (
            "For example, if a patient has T1=A, T1=B, T6=C, then the duration will be "
            "A: 2T, B: 3T, and the timeline from D1 will be A-A-B-B-B-C"
        )
        logger.warning(warn1 + warn2 + warn3)
    durees_divisees_par_deux = np.ceil(base_copy.loc[conditions, "evt_duration"].to_numpy() / 2)
    complementaire_des_durees = [
        tot - div_par_deux for tot, div_par_deux in zip(durees_totales, durees_divisees_par_deux, strict=False)
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
        raise ValueError(f"Problème dans les durées de traitements, regarder les patients {pat}") from exc

    # Mettre des NaN à la dernière ligne de chaque patient
    base_copy.loc[
        base_copy["ID_PATIENT"].ne(base_copy["ID_PATIENT"].shift(-1)),
        "evt_duration",
    ] = np.nan

    return base_copy


def _check_or_not(base: pd.DataFrame, check=True) -> pd.DataFrame:
    """Check base (presence de in, out, death, etc) et l'ordonne si check=True, sinon ne fait que l'ordonner.

    :param base: (df) une ligne = un évènement pour un patient, colonnes : 'ID_PATIENT', 'EVT' et 'TIMESTAMP'
    :param check: mettre à False si on veut ne pas passer par l'étape de check
    (par exemple si nous n'avons ni in ni out).
    :return: (df) copie de base réordonner (ID_PATIENT, TIMESTAMP)

    ..note: Attention, que check soit a True ou à False, base est réordonnée
    """
    base_copy = Checks(base).base if ("start" not in base["EVT"].unique() and check) else stable_sort(base)

    return base_copy
