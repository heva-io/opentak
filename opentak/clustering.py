from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy import cluster, spatial
from skimage.filters.rank import entropy
from skimage.morphology import square

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd


ID_PATIENT = str | int


def mean_spatial_entropy(arr: np.ndarray) -> float:
    """Compute the average entropy of array sub-blocks.

    Given a matrix of patients' time sequences as array, this function compute
    the probabilist entropy for each pixel on a 3*3 square and return the average.

    The minimum possible value for a single square is 0.0 (the matrix has a single
    value), and the maximum possible is ~2.20 if there are 9 or more possible values
    in the full matrix (a 3*3 square with 9 different values).

    :param arr: stacked patients' sequences
    :return: average entropy
    """
    mse: float = entropy(arr, square(3)).mean()
    return mse


class Tak:
    """Defines the TAK object."""

    def __init__(
        self,
        array: npt.NDArray,
        index_patients: npt.NDArray,
        dict_label_id: dict,
        timescale: int,
        evt_log: pd.DataFrame,
    ):
        """Initialise of the Tak class.

        :param array: 1 ligne = 1 patient, 1 column = 1 timestamp
        :param index_patients: patients IDs in the same order as the array matrix
        :param dict_label_id: dictionnaire mapping the name of teh event to its id.
        :param timescale: time windows size (in days) (resampling if !=1)
        :param evt_log: Initial base used by the tak.

        ..note: if index_patients[0] = 2, then 1st patient from array has ID 2
        """
        self.array = array

        self.dict_label_id = dict_label_id
        self.dict_label_id["other"] = 100
        self.homogeneity_base = mean_spatial_entropy(array)
        self.homogeneity_fitted: float
        self.index_patients = index_patients
        self.is_fitted = False
        self.list_ids_clusters: list[Any] = []
        self.sorted_array: list[Any] = []
        self.timescale = timescale
        self.base = evt_log

    def fit(self) -> Tak:
        """Fits the Tak object."""
        raise NotImplementedError

    def get_list_indices_cluster(self, list_ids_cluster: list | None = None):
        """Compute indices in list_ids_cluster corresponding to patients in list_ids_cluster

        :param list_ids_cluster: list of patient ids in the cluster format
        :return: list of arrays containing indices for each cluster
        """
        index_map = {pat: idx for idx, pat in enumerate(self.index_patients)}

        list_ids_cluster = (
            list_ids_cluster if list_ids_cluster is not None else self.list_ids_clusters
        )
        list_indices_cluster = [
            np.array([index_map[pat] for pat in cluster])
            for cluster in list_ids_cluster
        ]
        return list_indices_cluster

    def get_sorted_array(self, list_ids_cluster: list | None = None):
        """Compute the sorted array corresponding to list ids clusters

        :param list_ids_cluster: list of patient ids in the cluster format, if None uses self.list_ids_clusters
        :return: list of arrays containing sorted sequences for each cluster
        """
        list_indices_cluster = self.get_list_indices_cluster(list_ids_cluster)
        sorted_array = [
            self.array[indices_cluster] for indices_cluster in list_indices_cluster
        ]
        return sorted_array


class TakHca(Tak):
    """Classic TAK using hierarchical clustering."""

    def __init__(
        self,
        array: npt.NDArray,
        index_patients: npt.NDArray,
        dict_label_id: dict,
        timescale: int,
        evt_log: pd.DataFrame,
    ):
        """Initialize the TakHca class.

        :param array: 1 line = 1 patient, 1 column = 1 timestamp
        :param index_patients: patients IDs in the same order as the array matrix
        :param dict_label_id: dictionary mapping the name of the event to its id
        :param timescale: time windows size (in days) (resampling if !=1)
        :param evt_log: Initial base used by the tak
        """
        super().__init__(array, index_patients, dict_label_id, timescale, evt_log)
        self.pdist = None
        self.pdist_uncondensed: npt.NDArray | None = None
        self.distance: str | Callable = "hamming"
        self.method: str = "ward"
        self.linkage: npt.NDArray | None = None
        self.linkage_total: npt.NDArray | None = None

    def compute_pdist(
        self,
        distance: str | Callable = "hamming",
        subset_array: npt.NDArray | None = None,
    ) -> Tak:
        """Compute pairwise distance between patients' sequences.

        :param distance: Computation method for pairwise distance. Default to "Hamming"
        :param subset_array: subset of patients. If not provided the pairwise distance
        will be computed for all patients.
        :return: instance
        """
        if subset_array is None:
            subset_array = self.array

        self.distance = distance

        self.pdist = spatial.distance.pdist(subset_array, metric=self.distance)
        self.pdist_uncondensed = spatial.distance.squareform(self.pdist)
        return self

    def _check_pdist(self, pdist: npt.NDArray | None = None) -> npt.NDArray:
        # ruff: noqa: D205
        """Renvoie une erreur si la matrice pdist totale n'a pas été calculée,
        précédemment et n'est pas donné par l'utilisateur.

        :param pdist: matrice de pairwise distance entre les patients
        :return: pdist total si aucune n'était donnée en entrée, erreur si aucune
        n'était donnée et que pdist n'était pas calculée, pdist en entrée sinon.
        """
        # TODO remove redundant calls to this method, maybe even remove it altogether?
        if pdist is None:
            if self.pdist is not None:
                pdist = self.pdist
            else:
                raise AttributeError(
                    "You should compute a pdist first with .compute_pdist()"
                )
        return pdist

    def _get_linkage(
        self,
        method: str,
        pdist: npt.NDArray | None = None,
        optimal_ordering: bool = False,
    ) -> npt.NDArray:
        """Compute linkage matrix from pairwise distances.

        Calcule la matrice de linkage à partir de la matrice de distance pdist.

        :param method: linkage method ("ward", "single", "complete", "average")
        :param pdist: patients pairwise distances
        :param optimal_ordering: should reorder tree leaves?
        :return: linkage matrix
        """
        pdist = self._check_pdist(pdist)

        linkage = cluster.hierarchy.linkage(
            pdist,
            method=method,
            metric=self.distance,
            optimal_ordering=optimal_ordering,
        )

        return linkage

    def get_clusters(
        self,
        n_clusters: int = 1,
        method: str = "ward",
        patient_ids: Sequence | None = None,
        optimal_ordering: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clusters patients' sequences.

        :param n_clusters: number of clusters to create
        :param method: linkage method ("ward", "single", "complete", "average")
        :param patient_ids: list of patients' ids to cluster, if None, all patients are used
        :param optimal_ordering: reorder tree leaves (longer computation time)
        :return: tuple of (cluster number for each patient, list of patient indices in optimal order)
        """
        if patient_ids is None:
            # get computed distance matrix if already done
            pdist = self._check_pdist(pdist=None)
        else:
            if len(patient_ids) < n_clusters:
                raise ValueError(
                    f"Cannot compute more clusters than patients {len(patient_ids)} "
                    f"were provided for {n_clusters} clusters"
                )
            if len(patient_ids) == 1:
                return np.array([0]), np.array([0])

            # if a list of patient is provided, the pdist is already computed
            if self.pdist_uncondensed is None:
                raise ValueError(
                    "self.pdist_uncondensed is None, call self.compute_pdist() first"
                )
            pdist_uncondensed_ids = self.pdist_uncondensed[patient_ids][:, patient_ids]
            pdist = spatial.distance.squareform(pdist_uncondensed_ids)
        linkage = self._get_linkage(
            pdist=pdist, method=method, optimal_ordering=optimal_ordering
        )

        if patient_ids is None or len(patient_ids) == len(self.index_patients):
            self.linkage_total = linkage

        list_indices_ordered = cluster.hierarchy.leaves_list(linkage)

        patients_groups_id = cluster.hierarchy.cut_tree(linkage, n_clusters=n_clusters)[
            :, 0
        ]

        return patients_groups_id, list_indices_ordered

    def fit(
        self,
        n_clusters: int = 1,
        method: str = "ward",
        distance: str | Callable = "hamming",
        optimal_ordering: bool = True,
        global_optimal_ordering: bool = False,
    ) -> Tak:
        """Cluster patients' sequences.

        ```
        Shorthand method for:
        1. Computing pairwise distances
        2. Building linkage matrix
        3. Ordering patients' sequences
        ```

        !!! note "Concept"
            Regarder les feuilles du dendrogram (les patients) dans l'ordre et avoir un array où les patients
            similaires sont cote à cote.

        !!! note
            list_leaves contient autant d'éléments que de clusters demandés, chaque élément étant la liste des ID
            des patients dans le cluster

        :param n_clusters: number of clusters to create
        :param method: linkage method ("ward", "single", "complete", "average")
        :param distance: pairwise distance method
        :param optimal_ordering: should reorder tree leaves?
        :param global_optimal_ordering: whether to perform optimal ordering on all patients globally
        :return: self for method chaining
        """
        is_pdist_obsolete = self.pdist is None or (distance, method) != (
            self.distance,
            self.method,
        )

        if is_pdist_obsolete:
            self.compute_pdist(distance)

        # perform HCA clustering
        labels, list_indices = self.get_clusters(
            n_clusters=n_clusters,
            method=method,
            optimal_ordering=global_optimal_ordering,
        )

        list_ids_clusters = []

        for label in sorted(np.unique(labels)):
            # Get patients index for cluster
            list_idx_cluster = np.arange(len(labels))[labels == label]

            # Order of patients in cluster, from 0 to len(cluster)
            if not global_optimal_ordering:
                # if optimal ordering is not perform on all patients, perform it individually on each cluster
                _, list_order_pat_cluster = self.get_clusters(
                    n_clusters=1,
                    method=method,
                    patient_ids=list_idx_cluster,
                    optimal_ordering=optimal_ordering,
                )
            else:
                # retrieve order from global order
                cluster_positions = {
                    idx: pos for pos, idx in enumerate(list_idx_cluster)
                }
                # position of base labels in sorted labels
                list_order_pat_cluster = np.array(
                    [
                        cluster_positions[idx]
                        for idx in list_indices
                        if idx in cluster_positions
                    ]
                )

            # Get corresponding patient ids
            list_id_index_cluster = self.index_patients[labels == label]
            # sort these patients according to the optimal ordering
            list_id_index_cluster_ordered = list_id_index_cluster[
                list_order_pat_cluster
            ]

            # Save results for this cluster in a list,
            # to go on with the next cluster
            list_ids_clusters.append(list_id_index_cluster_ordered)

        # ruff noqa: ERA001 # self.homogeneity_fitted = _homogeneity_fitted...

        self.list_ids_clusters = list_ids_clusters
        self.sorted_array = self.get_sorted_array()
        self.is_fitted = True
        return self
