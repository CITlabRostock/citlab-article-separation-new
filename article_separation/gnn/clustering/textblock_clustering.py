import logging
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import dbscan
from scipy.stats import gmean
from gnn.clustering.dbscan import DBScanRelation


class TextblockClustering(object):
    """
    Class for textblock clustering.

    Usage::

        tb_clustering = TextblockClustering(flags)
        …
        tb_clustering.set_confs(confs, symmetry_fn=scipy.stats.gmean)
        tb_clustering.calc(method='greedy')
        print(tb_clustering.num_classes, tb_clustering.num_noise)
        print(tb_clustering.tb_labels, tb_clustering.tb_classes)
        print(tb_clustering.rel_LLH)

    """

    def __init__(self, flags):
        self.clustering_params = dict()
        self._flags = flags

        # Default params which are method dependant
        # [dbscan]
        self.clustering_params["min_neighbors_for_cluster"] = 1
        self.clustering_params["confidence_threshold"] = 0.5
        self.clustering_params["cluster_agreement_threshold"] = 0.5
        self.clustering_params["assign_noise_clusters"] = True
        # [linkage]
        self.clustering_params["method"] = "centroid"
        self.clustering_params["criterion"] = "distance"
        self.clustering_params["t"] = -1.0
        # [greedy]
        self.clustering_params["max_iteration"] = 1000
        # [dbscan_std]
        self.clustering_params["epsilon"] = 0.5
        self.clustering_params["min_samples"] = 1

        # Updating of the default params if provided via flags as a dict
        for i in self._flags.clustering_params:
            if i not in self.clustering_params:
                logging.critical(f"Given input_params-key '{i}' is not used by class 'TextblockClustering'!")
        self.clustering_params.update(flags.clustering_params)

        # Public params to access clustering results
        self.tb_labels = None  # list of int: textblock labels
        self.tb_classes = None  # list of list of int: textblock classes
        self.num_classes = 0  # int: counts classes
        self.num_noise = 0  # int: counts noise classes (see DBSCAN algorithm)
        self.rel_LLH = 0.0  # float: special relative-loglikelihood value (for internal comparison only)

        # Private params for internal computations
        self._conf_mat = None
        self._mat_dim = None
        self._dist_mat = None
        self._cond_dists = None
        self._delta_mat = None
        self._dbscanner = None

    #     # Debug
    #     self._page_path = None
    #     self._save_dir = None
    #     self._debug = False
    #
    # def set_debug(self, page_path, save_dir):
    #     self._page_path = page_path
    #     self._save_dir = save_dir
    #     self._debug = True

    def print_params(self):
        logging.info("CLUSTERING:")
        sorted_dict = sorted(self.clustering_params.items(), key=lambda kv: kv[0])
        for a in sorted_dict:
            logging.info(f"  {a[0]}: {a[1]}")

    def get_info(self, method):
        """Returns an info string with the most important paramters of the given method"""
        info_string = None
        if getattr(self, f'_{method}', None):
            if method == 'dbscan':
                info_string = f'dbscan_conf{self.clustering_params["confidence_threshold"]}_' \
                              f'cluster{self.clustering_params["cluster_agreement_threshold"]}'
            elif method == 'dbscan_std':
                info_string = f'dbscan_std_eps{self.clustering_params["epsilon"]}_' \
                              f'samples{self.clustering_params["min_samples"]}'
            elif method == 'linkage':
                info_string = f'linkage_{self.clustering_params["method"]}_{self.clustering_params["criterion"]}_' \
                              f't{self.clustering_params["t"]}'
            elif method == 'greedy':
                info_string = f'greedy_iter{self.clustering_params["max_iteration"]}'
        return info_string

    def set_confs(self, confs, symmetry_fn=gmean):
        """
        Sets confidence values and symmetrization function.

        Note that since the `symmetry_fn` will be applied to confidences, the geometric mean is preferred over
        the arithmetic mean. But suitable is any function element-wise applicable to 2D-arrays.

        :param confs: confidence array with values from (0,1)
        :param symmetry_fn: array-like function to make confidences symmetric
        :return: None
        """
        self._conf_mat = np.array(confs)
        self._mat_dim = self._conf_mat.shape[0]
        # Substitute confidences of 0.0 and 1.0 with next bigger/smaller float (to prevent zero divides)
        self._smooth_confs()
        # Make confidence matrix symmetric (if not already is)
        if symmetry_fn:
            self._make_symmetric(symmetry_fn)
        # Convert to (pseudo) distances
        self._dist_mat = -np.log(self._conf_mat)
        # Distances for same elements should be 0
        np.fill_diagonal(self._dist_mat, 0.0)
        #   linkage (condensed distance array)
        cond_indices = np.triu_indices_from(self._dist_mat, k=1)
        self._cond_dists = self._dist_mat[cond_indices]
        #   greedy
        self._delta_mat = np.array(list(map(lambda p: np.log(p / (1 - p)), self._conf_mat)))
        np.fill_diagonal(self._delta_mat, -np.math.inf)

    def _make_symmetric(self, symmetry_fn):
        mat = self._conf_mat
        mat_transpose = self._conf_mat.transpose()
        temp_mat = np.stack([mat, mat_transpose], axis=-1)
        self._conf_mat = symmetry_fn(temp_mat, axis=-1)

    def _smooth_confs(self):
        dtype = self._conf_mat.dtype
        min_val = np.nextafter(0, 1, dtype=dtype)
        max_val = np.nextafter(1, 0, dtype=dtype)
        self._conf_mat[self._conf_mat == 0.0] = min_val
        self._conf_mat[self._conf_mat == 1.0] = max_val

    def calc(self, method):
        """
        Run calculation of clusters.

        Currently implemented methods: 'dbscan', 'linkage', 'greedy', 'dbscan_std'

        :param method: method name (str)
        :return: None
        """
        self.tb_labels = None
        self.tb_classes = None

        if self._mat_dim == 2:  # we have exactly two text regions
            logging.info(f'No clustering performed for two text regions. Decision based on confidence '
                         f'threshold ({self._conf_mat[0, 1]} >= {self.clustering_params["confidence_threshold"]}).')
            self.tb_labels = [1, 1] if self._conf_mat[0, 1] >= self.clustering_params["confidence_threshold"] else [1,
                                                                                                                    2]
        else:  # atleast three text regions
            if getattr(self, f'_{method}', None):
                fctn = getattr(self, f'_{method}', None)
                logging.info(f'performing clustering with method "{method}"')
                fctn()
            else:
                raise NotImplementedError(f'cannot find clustering method "_{method}"!')
        self._calc_relative_LLH()

    def _labels2classes(self):
        class_dict = {}
        for (tb, cls) in enumerate(self.tb_labels):
            if class_dict.get(cls, None):
                c = class_dict.get(cls, None)
                c.append(tb)
            else:
                class_dict[cls] = [tb]
        self.tb_classes = list(map(sorted, class_dict.values()))

    def _classes2labels(self):
        self.tb_labels = np.empty(self._mat_dim, dtype=int)
        self.tb_labels.fill(-1)
        for (idx, cls) in enumerate(self.tb_classes):
            for tb in cls:
                self.tb_labels[tb] = idx

    def _calc_relative_LLH(self):
        self.rel_LLH = 0.0
        for idx0 in range(self._mat_dim):
            if self.tb_labels[idx0] >= 0:
                for idx1 in range(idx0):
                    if self.tb_labels[idx0] == self.tb_labels[idx1]:
                        delta_LLH = (self._delta_mat[idx0, idx1] + self._delta_mat[idx1, idx0]) / 2
                        self.rel_LLH += delta_LLH

    def _dbscan_std(self):
        (_, self.tb_labels) = dbscan(self._dist_mat,
                                     metric='precomputed',
                                     min_samples=self.clustering_params["min_samples"],
                                     eps=self.clustering_params["epsilon"])
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _greedy(self):
        self.tb_labels = np.array(range(self._mat_dim), dtype=int)
        self._labels2classes()
        self._calcMat = self._delta_mat.copy()
        iter_count = self.clustering_params["max_iteration"]
        while iter_count > 0:
            iter_count -= 1
            #  stärkste Kante
            (i, j) = np.unravel_index(np.argmax(self._calcMat), self._conf_mat.shape)
            if self._calcMat[i, j] > 0:
                logging.debug(f'{i}–{j} mit {self._calcMat[i, j]} …')
                self._greedy_step(i, j)
                self._classes2labels()
                self._calc_relative_LLH()
                logging.debug(f'… LLH = {self.rel_LLH}')
                logging.debug(f'{self.tb_labels}')
                logging.debug(f'{self.tb_classes}')
            else:
                logging.info(f'… after {self.clustering_params["max_iteration"] - iter_count} iterations')
                break

        self.tb_classes = [cls for cls in self.tb_classes if len(cls) > 0]
        self.num_classes = len(self.tb_classes)
        self._classes2labels()
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _greedy_step(self, cls0, cls1):
        self.tb_classes[cls0].extend(self.tb_classes[cls1])
        self.tb_classes[cls0] = sorted(self.tb_classes[cls0])
        self.tb_classes[cls1] = []

        for idx in range(self._mat_dim):
            if idx != cls0 and idx != cls1:
                self._calcMat[idx, cls0] += self._calcMat[idx, cls1]
                self._calcMat[cls0, idx] = self._calcMat[idx, cls0]

        for idx in range(self._mat_dim):
            self._calcMat[idx, cls1] = -np.math.inf
            self._calcMat[cls1, idx] = self._calcMat[idx, cls1]

    def _linkage(self):
        linkage_res = linkage(self._cond_dists, method=self.clustering_params["method"])

        t = self.clustering_params["t"]
        if t < 0:
            hierarchical_distances = linkage_res[:, 2]
            distance_mean = float(np.mean(hierarchical_distances))
            distance_median = float(np.median(hierarchical_distances))
            t = 1 / 2 * (distance_mean + distance_median)

        self.tb_labels = fcluster(linkage_res, t=t, criterion=self.clustering_params["criterion"])
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _dbscan(self):
        if not self._dbscanner:
            self._dbscanner = DBScanRelation(
                min_neighbors_for_cluster=self.clustering_params["min_neighbors_for_cluster"],
                confidence_threshold=self.clustering_params["confidence_threshold"],
                cluster_agreement_threshold=self.clustering_params["cluster_agreement_threshold"],
                assign_noise_clusters=self.clustering_params["assign_noise_clusters"])

        self.tb_labels = self._dbscanner.cluster_relations(self._mat_dim, self._conf_mat)
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])
