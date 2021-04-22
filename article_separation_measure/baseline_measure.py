# -*- coding: utf-8 -*-

import numpy as np


class BaselineMeasureResult(object):
    def __init__(self):
        self.page_wise_per_dist_tol_tick_per_line_recall = []
        self.page_wise_per_dist_tol_tick_recall = []
        self.page_wise_recall = []
        self.recall = 0.0
        self.page_wise_per_dist_tol_tick_per_line_precision = []
        self.page_wise_per_dist_tol_tick_precision = []
        self.page_wise_precision = []
        self.precision = []


class BaselineMeasure(object):
    def __init__(self):
        self.result = BaselineMeasureResult()

    def add_per_dist_tol_tick_per_line_recall(self, per_dist_tol_tick_per_line_recall):
        """ #distTolTicks x #truthBaseLines matrix of recalls, stores results """
        assert type(per_dist_tol_tick_per_line_recall) == np.ndarray,\
            "per_dist_tol_tick_per_line_recall has to be np.ndarray"
        assert len(per_dist_tol_tick_per_line_recall.shape) == 2,\
            "per_dist_tol_tick_per_line_recall has to be 2d"
        assert per_dist_tol_tick_per_line_recall.dtype == float,\
            "per_dist_tol_tick_per_line_recall has to be float"

        # page wise recall: per tol, per line
        self.result.page_wise_per_dist_tol_tick_per_line_recall.append(per_dist_tol_tick_per_line_recall)

        # page wise recall: per tol (summed over lines)
        per_dist_tol_tick_recall = np.sum(per_dist_tol_tick_per_line_recall, axis=1)
        per_dist_tol_tick_recall /= per_dist_tol_tick_per_line_recall.shape[1]
        self.result.page_wise_per_dist_tol_tick_recall.append(per_dist_tol_tick_recall)

        # page wise recall: summed over tols & lines
        recall = np.sum(per_dist_tol_tick_recall)
        recall /= per_dist_tol_tick_recall.shape[0]
        self.result.page_wise_recall.append(recall)

        # TODO: We don't need that here for articles, since we use the article_wise_recall (here page_wise)
        # TODO: matrix for a greedy alignment first (-> remove for speedup?)
        self.calc_recall()

    def add_per_dist_tol_tick_per_line_precision(self, per_dist_tol_tick_per_line_precision):
        """ #distTolTicks x #recoBaseLines matrix of precisions, stores results"""
        assert type(per_dist_tol_tick_per_line_precision) == np.ndarray,\
            "per_dist_tol_tick_per_line_precision has to be np.ndarray"
        assert len(per_dist_tol_tick_per_line_precision.shape) == 2,\
            "per_dist_tol_tick_per_line_precision has to be 2d matrix"
        assert per_dist_tol_tick_per_line_precision.dtype == float,\
            "per_dist_tol_tick_per_line_precision has to be float"

        # page wise precision: per tol, per line
        self.result.page_wise_per_dist_tol_tick_per_line_precision.append(per_dist_tol_tick_per_line_precision)

        # page wise precision: per tol (summed over lines)
        per_dist_tol_tick_precision = np.sum(per_dist_tol_tick_per_line_precision, axis=1)
        per_dist_tol_tick_precision /= per_dist_tol_tick_per_line_precision.shape[1]
        self.result.page_wise_per_dist_tol_tick_precision.append(per_dist_tol_tick_precision)

        # page wise precision: summed over tols & lines
        precision = np.sum(per_dist_tol_tick_precision)
        precision /= per_dist_tol_tick_precision.shape[0]
        self.result.page_wise_precision.append(precision)

        # TODO: We don't need that here for articles, since we use the article_wise_precision (here page_wise)
        # TODO: matrix for a greedy alignment first (-> remove for speedup?)
        self.calc_precision()

    def calc_recall(self):
        """ average recall over all pages and store result """
        avg_recall = 0.0
        for recall in self.result.page_wise_recall:
            avg_recall += recall
        avg_recall /= len(self.result.page_wise_recall)
        self.result.recall = avg_recall

    def calc_precision(self):
        """ average precision over all pages and store result """
        avg_precision = 0.0
        for precision in self.result.page_wise_precision:
            avg_precision += precision
        avg_precision /= len(self.result.page_wise_precision)
        self.result.precision = avg_precision

    # no usage
    def get_page_wise_true_false_counts_hypo(self, threshold):
        assert type(threshold) == float, "threshold has to be float"

        true_false_positives = np.zeros([2, len(self.result.page_wise_per_dist_tol_tick_per_line_precision)])

        for i, per_dist_tol_tick_per_line_precision in \
                enumerate(self.result.page_wise_per_dist_tol_tick_per_line_precision):
            avg_per_line_precision = np.sum(per_dist_tol_tick_per_line_precision, axis=0)
            avg_per_line_precision /= per_dist_tol_tick_per_line_precision.shape[0]

            true_pos = (avg_per_line_precision >= threshold).as_type(np.float32)
            false_pos = 1.0 - true_pos
            true_false_positives[0, i] = np.sum(true_pos)
            true_false_positives[1, i] = np.sum(false_pos)

        return true_false_positives

    # no usage
    def get_page_wise_true_false_counts_gt(self, threshold):
        assert type(threshold) == float, "threshold has to be float"

        true_false_negatives = np.zeros([2, len(self.result.page_wise_per_dist_tol_tick_per_line_recall)])

        for i, per_dist_tol_tick_per_line_recall in enumerate(self.result.page_wise_per_dist_tol_tick_per_line_recall):
            avg_per_line_recall = np.sum(per_dist_tol_tick_per_line_recall, axis=0)
            avg_per_line_recall /= per_dist_tol_tick_per_line_recall.shape[0]

            true_neg = (avg_per_line_recall >= threshold).as_type(np.float32)
            false_neg = 1.0 - true_neg
            true_false_negatives[0, i] = np.sum(true_neg)
            true_false_negatives[1, i] = np.sum(false_neg)

        return true_false_negatives

    # no usage
    def get_specific_page_true_false_constellation(self, page_num, threshold):
        assert type(page_num) == int, "page_num has to be int"
        assert type(threshold) == float, "threshold has to be float"

        per_dist_tol_tick_per_line_recall = self.result.page_wise_per_dist_tol_tick_per_line_recall[page_num]
        avg_per_line_recall = np.sum(per_dist_tol_tick_per_line_recall, axis=0)
        avg_per_line_recall /= per_dist_tol_tick_per_line_recall.shape[0]
        gt_correct = avg_per_line_recall >= threshold

        per_dist_tol_tick_per_line_precision = self.result.page_wise_per_dist_tol_tick_per_line_precision[page_num]
        avg_per_line_precision = np.sum(per_dist_tol_tick_per_line_precision, axis=0)
        avg_per_line_precision /= per_dist_tol_tick_per_line_precision.shape[0]
        hypo_correct = avg_per_line_precision >= threshold

        return np.concatenate((np.expand_dims(gt_correct, axis=0), np.expand_dims(hypo_correct, axis=0)), axis=0)
