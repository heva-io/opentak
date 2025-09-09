from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt
from scipy import cluster, spatial

if TYPE_CHECKING:
    from collections.abc import Sequence

    import pandas as pd
    from scipy.spatial.distance import _Metric

    LinkageMethod = Literal[
        "single", "complete", "average", "weighted", "centroid", "median", "ward"
    ]


ID_PATIENT = str | int


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
        """Initialize the Tak class.

        :param array: 1 row = 1 patient, 1 column = 1 timestamp
        :param index_patients: patient IDs in the same order as in ``array``
        :param dict_label_id: dictionary mapping event names to their IDs
        :param timescale: time window size (in days); sequences may be resampled if ``!= 1``
        :param evt_log: initial event log used by TAK
        """
        self.array: npt.NDArray = array

        self.dict_label_id = dict_label_id
        self.dict_label_id["other"] = 100
        self.index_patients = index_patients
        self.is_fitted = False
        self.list_ids_clusters: list[Any] = []
        self.sorted_array: list[Any] = []
        self.timescale = timescale
        self.base = evt_log

    def fit(
        self,
    ) -> Tak:
        """Fit the TAK model (to be implemented by subclasses)."""
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
        self.pdist: npt.NDArray | None = None
        self.pdist_uncondensed: npt.NDArray | None = None
        self.distance: _Metric = "hamming"
        self.method: LinkageMethod = "ward"
        self.linkage: npt.NDArray | None = None
        self.linkage_total: npt.NDArray | None = None

    def compute_pdist(
        self,
        distance: _Metric = "hamming",
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
        """Raise if the global pdist wasn't computed earlier and isn't provided by user.

        :param pdist: pairwise distance matrix between patients
        :return: the global ``pdist`` when none is provided, or the provided ``pdist``; raises if neither is available
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
        method: LinkageMethod,
        pdist: npt.NDArray | None = None,
        optimal_ordering: bool = True,
    ) -> npt.NDArray:
        """Compute the linkage matrix from the pairwise distance vector/matrix.

        :param method: linkage method ("ward", "single", "complete", "average", ...)
        :param pdist: patients' pairwise distances
        :param optimal_ordering: whether to reorder tree leaves for optimal ordering
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
        method: LinkageMethod = "ward",
        patient_ids: Sequence | None = None,
        optimal_ordering: bool = True,
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

        patients_groups_id = cluster.hierarchy.cut_tree(
            linkage, n_clusters=np.array([n_clusters])
        )[:, 0]

        return patients_groups_id, list_indices_ordered

    def fit(
        self,
        n_clusters: int = 1,
        method: LinkageMethod = "ward",
        distance: _Metric = "hamming",
        optimal_ordering: bool = True,
    ) -> Tak:
        """Cluster patients' sequences.

        Shorthand for:
        1. Computing pairwise distances
        2. Building the linkage matrix
        3. Ordering patients by dendrogram leaves

        :param n_clusters: number of clusters to create
        :param method: linkage method ("ward", "single", "complete", "average")
        :param distance: pairwise distance method
        :param optimal_ordering: whether to reorder tree leaves (optimal ordering)
        :return: TAK fitted
        """
        is_pdist_obsolete = self.pdist is None or (distance, method) != (
            self.distance,
            self.method,
        )

        if is_pdist_obsolete:
            self.compute_pdist(distance)

        patient_cluster_labels, list_indices = self.get_clusters(
            n_clusters=n_clusters, method=method, optimal_ordering=optimal_ordering
        )

        cluster_labels_ordered = patient_cluster_labels[list_indices]
        cluster_order_by_leaves = list(dict.fromkeys(cluster_labels_ordered.tolist()))
        list_ids_ordered = [
            list_indices[cluster_labels_ordered == c].tolist()
            for c in cluster_order_by_leaves
        ]
        list_ids_cluster_ordered = [
            [self.index_patients[i] for i in group] for group in list_ids_ordered
        ]

        self.list_ids_clusters = list_ids_cluster_ordered
        self.sorted_array = self.get_sorted_array()
        self.is_fitted = True
        return self
