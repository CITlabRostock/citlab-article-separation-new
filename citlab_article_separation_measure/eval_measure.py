# -*- coding: utf-8 -*-

import jpype
import numpy as np
from citlab_python_util.geometry.polygon import norm_poly_dists, Polygon
from citlab_python_util.geometry.util import calc_tols

from citlab_article_separation_measure.baseline_measure import BaselineMeasure


class BaselineMeasureEval(object):
    def __init__(self, min_tol=10, max_tol=30, rel_tol=0.25, poly_tick_dist=5):
        """ Initialize BaselineMeasureEval object.

        :param min_tol: MINIMUM distance tolerance which is not penalized
        :param max_tol: MAXIMUM distance tolerance which is not penalized
        :param rel_tol: fraction of estimated interline distance as tolerance values
        :param poly_tick_dist: desired distance of points of the baseline
        """
        assert type(min_tol) == int and type(max_tol) == int, "min_tol and max_tol have to be ints"
        assert min_tol <= max_tol, "min_tol can't exceed max_tol"
        assert 0.0 < rel_tol <= 1.0, "rel_tol has to be in the range (0,1]"
        assert type(poly_tick_dist) == int, "poly_tick_dist has to be int"

        self.max_tols = np.arange(min_tol, max_tol + 1)
        self.rel_tol = rel_tol
        self.poly_tick_dist = poly_tick_dist
        self.truth_line_tols = None
        self.measure = BaselineMeasure()

    def calc_measure_for_page_baseline_polys(self, polys_truth, polys_reco, use_java_code=True):
        """ Calculates the BaselinMeasure stats for given truth and reco polygons of a single page and adds the results
        to the BaselineMeasure structure.

        NOTE: We can choose between the usage of java (much more faster!!!) or python methods.

        :param polys_truth: list of TRUTH polygons corresponding to a single page
        :param polys_reco: list of RECO polygons corresponding to a single page
        :param use_java_code: usage of methods written in java or not
        """
        assert type(polys_truth) == list and type(polys_reco) == list, "polys_truth and polys_reco have to be lists"
        assert all([isinstance(poly, Polygon) for poly in polys_truth + polys_reco]), \
            "elements of polys_truth and polys_reco have to be Polygons"

        # call java code to execute the method
        if use_java_code:
            polys_truth_java = []
            polys_reco_java = []

            for poly in polys_truth:
                polys_truth_java.append(jpype.java.awt.Polygon(poly.x_points, poly.y_points, poly.n_points))
            for poly in polys_reco:
                polys_reco_java.append(jpype.java.awt.Polygon(poly.x_points, poly.y_points, poly.n_points))

            java_object = jpype.JPackage("citlab_article_separation_measure.external.java").Util()

            pr_list = \
                java_object.calcMetricForPageBaseLinePolys(polys_truth_java, polys_reco_java,
                                                           self.max_tols.tolist(), self.poly_tick_dist, self.rel_tol)

            precision = np.array([list(pr_list[0][i]) for i in range(len(pr_list[0]))])
            recall = np.array([list(pr_list[1][i]) for i in range(len(pr_list[0]))])

        # call python code to execute the method
        else:
            # Normalize baselines, so that poly points have a desired "distance"
            polys_truth_norm = norm_poly_dists(polys_truth, self.poly_tick_dist)
            polys_reco_norm = norm_poly_dists(polys_reco, self.poly_tick_dist)

            # Optionally calculate tolerances
            if self.max_tols[0] < 0:
                # call python class to calculate the tolerances
                tols = calc_tols(polys_truth_norm, self.poly_tick_dist, 250, self.rel_tol)
                self.truth_line_tols = np.expand_dims(tols, axis=1)
            else:
                self.truth_line_tols = np.tile(self.max_tols, [len(polys_truth_norm), 1]).astype(float)

            # For each reco poly calculate the precision values for all tolerances
            precision = self.calc_precision(polys_truth_norm, polys_reco_norm)
            # For each truth_poly calculate the recall values for all tolerances
            recall = self.calc_recall(polys_truth_norm, polys_reco_norm)

        # add results
        self.measure.add_per_dist_tol_tick_per_line_precision(precision)
        self.measure.add_per_dist_tol_tick_per_line_recall(recall)
        self.truth_line_tols = None

    def calc_precision(self, polys_truth, polys_reco):
        """ Calculates and returns precision values for given truth and reco polygons for all tolerances.

        NOTE: this method will be redundant if we use the java code block in "calc_measure_for_page_baseline_polys"

        :param polys_truth: list of TRUTH polygons
        :param polys_reco: list of RECO polygons
        :return: precision values
        """
        assert type(polys_truth) == list and type(polys_reco) == list, "polys_truth and polys_reco have to be lists"
        assert all([isinstance(poly, Polygon) for poly in polys_truth + polys_reco]), \
            "elements of polys_truth and polys_reco have to be Polygons"

        # relative hits per tolerance value over all reco and truth polygons
        rel_hits = np.zeros([self.max_tols.shape[0], len(polys_reco), len(polys_truth)])

        for i, poly_reco in enumerate(polys_reco):
            for j, poly_truth in enumerate(polys_truth):
                rel_hits[:, i, j] = self.count_rel_hits(poly_reco, poly_truth, self.truth_line_tols[j])

        # calculate alignment
        precision = np.zeros([self.max_tols.shape[0], len(polys_reco)])
        for i, hits_per_tol in enumerate(np.split(rel_hits, rel_hits.shape[0])):
            hits_per_tol = np.squeeze(hits_per_tol, 0)
            while True:
                # calculate indices for maximum alignment
                max_idx_x, max_idx_y = np.unravel_index(np.argmax(hits_per_tol), hits_per_tol.shape)
                # finish if all polys_reco have been aligned
                if hits_per_tol[max_idx_x, max_idx_y] < 0:
                    break
                # set precision to max alignment
                precision[i, max_idx_x] = hits_per_tol[max_idx_x, max_idx_y]
                # set row and column to -1
                hits_per_tol[max_idx_x, :] = -1.0
                hits_per_tol[:, max_idx_y] = -1.0

        return precision

    def count_rel_hits(self, poly_to_count, poly_ref, tols):
        """ Counts the relative hits per tolerance value over all points of the polygon and corresponding nearest points
        of the reference polygon.

        NOTE: this method will be redundant if we use the java code block in "calc_measure_for_page_baseline_polys"

        :param poly_to_count: Polygon to count over
        :param poly_ref: reference Polygon
        :param tols: vector of tolerances
        :return: vector of relative hits for every tolerance value
        """
        assert isinstance(poly_to_count, Polygon) and isinstance(poly_ref, Polygon), \
            "poly_to_count and poly_ref have to be Polygons"
        assert type(tols) == np.ndarray, "tols has to be np.ndarray"
        assert len(tols.shape) == 1, "tols has to be 1d vector"
        assert tols.dtype == float, "tols has to be float"

        poly_to_count_bb = poly_to_count.get_bounding_box()
        poly_ref_bb = poly_ref.get_bounding_box()
        intersection = poly_to_count_bb.intersection(poly_ref_bb)
        rel_hits = np.zeros_like(tols)

        # Early stopping criterion
        if min(intersection.width, intersection.height) < -3.0 * tols[-1]:
            return rel_hits

        # Build and expand numpy arrays from points
        poly_to_count_x = np.array(poly_to_count.x_points)
        poly_to_count_y = np.array(poly_to_count.y_points)
        poly_ref_x = np.expand_dims(np.asarray(poly_ref.x_points), axis=1)
        poly_ref_y = np.expand_dims(np.asarray(poly_ref.y_points), axis=1)

        # Calculate minimum distances
        dist_x = abs(poly_to_count_x - poly_ref_x)
        dist_y = abs(poly_to_count_y - poly_ref_y)
        min_dist = np.amin(dist_x + dist_y, axis=0)

        # Calculate masks for two tolerance cases
        tols_t = np.expand_dims(np.asarray(tols), axis=1)

        mask1 = (min_dist <= tols_t).astype(float)
        mask2 = (min_dist <= 3.0 * tols_t).astype(float)
        mask2 = mask2 - mask1

        # Calculate relative hits
        rel_hits = mask1 + mask2 * ((3.0 * tols_t - min_dist) / (2.0 * tols_t))
        rel_hits = np.sum(rel_hits, axis=1)

        rel_hits /= poly_to_count.n_points
        return rel_hits

    def calc_recall(self, polys_truth, polys_reco):
        """ Calculates and returns recall values for given truth and reco polygons for all tolerances.

        NOTE: this method will be redundant if we use the java code block in "calc_measure_for_page_baseline_polys"

        :param polys_truth: list of TRUTH polygons
        :param polys_reco: list of RECO polygons
        :return: recall values
        """
        assert type(polys_truth) == list and type(polys_reco) == list, "polys_truth and polys_reco have to be lists"
        assert all([isinstance(poly, Polygon) for poly in polys_truth + polys_reco]), \
            "elements of polys_truth and polys_reco have to be Polygons"

        recall = np.zeros([self.max_tols.shape[0], len(polys_truth)])
        for i, poly_truth in enumerate(polys_truth):
            recall[:, i] = self.count_rel_hits_list(poly_truth, polys_reco, self.truth_line_tols[i])

        return recall

    def count_rel_hits_list(self, poly_to_count, polys_ref, tols):
        """
        NOTE: this method will be redundant if we use the java code block in "calc_measure_for_page_baseline_polys"

        :param poly_to_count: Polygon to count over
        :param polys_ref: list of reference Polygons
        :param tols: vector of tolerances
        :return:
        """
        assert isinstance(poly_to_count, Polygon), "poly_to_count has to be Polygon"
        assert type(polys_ref) == list, "polys_ref has to be list"
        assert all([isinstance(poly, Polygon) for poly in polys_ref]), "elements of polys_ref have to Polygons"
        assert type(tols) == np.ndarray, "tols has to be np.ndarray"
        assert len(tols.shape) == 1, "tols has to be 1d vector"
        assert tols.dtype == float, "tols has to be float"

        poly_to_count_bb = poly_to_count.get_bounding_box()

        all_inf = True
        min_dist = np.full((poly_to_count.n_points,), np.inf)

        for poly_ref in polys_ref:
            poly_ref_bb = poly_ref.get_bounding_box()
            intersection = poly_to_count_bb.intersection(poly_ref_bb)

            # Early stopping criterion
            if min(intersection.width, intersection.height) < -3.0 * tols[-1]:
                continue

            # Build and expand numpy arrays from points
            poly_to_count_x = np.array(poly_to_count.x_points)
            poly_to_count_y = np.array(poly_to_count.y_points)
            poly_ref_x = np.expand_dims(np.asarray(poly_ref.x_points), axis=1)
            poly_ref_y = np.expand_dims(np.asarray(poly_ref.y_points), axis=1)

            # Calculate minimum distances
            dist_x = abs(poly_to_count_x - poly_ref_x)
            dist_y = abs(poly_to_count_y - poly_ref_y)

            if all_inf:
                all_inf = False
                min_dist = np.amin(dist_x + dist_y, axis=0)
            else:
                min_dist = np.minimum(min_dist, np.amin(dist_x + dist_y, axis=0))

        # Calculate masks for two tolerance cases
        tols_t = np.expand_dims(np.asarray(tols), axis=1)

        mask1 = (min_dist <= tols_t).astype(float)
        mask2 = (min_dist <= 3.0 * tols_t).astype(float)
        mask2 = mask2 - mask1

        # Calculate relative hits
        if all_inf:
            rel_hits = np.zeros(mask1.shape)
        else:
            rel_hits = mask1 + mask2 * ((3.0 * tols_t - min_dist) / (2.0 * tols_t))
            rel_hits = np.nan_to_num(rel_hits)

        rel_hits = np.sum(rel_hits, axis=1)

        rel_hits /= poly_to_count.n_points
        return rel_hits
