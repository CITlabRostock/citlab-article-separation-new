# -*- coding: utf-8 -*-

import jpype
import numpy as np
import os
from argparse import ArgumentParser
from citlab_python_util.parser.xml.page.page import Page

from citlab_python_util.math.measure import f_measure
from citlab_article_separation_measure.eval_measure import BaselineMeasureEval


def get_data_from_pagexml(path_to_pagexml):
    """
    :param path_to_pagexml: file path

    :return: dictionary with the article / block ID's as keys and a list of corresponding baselines (given by polygons)
    as values
    """
    art_polygons_dict = {}

    try:
        # load the page xml file
        page_file = Page(path_to_xml=path_to_pagexml)
        # get all text lines article wise
        art_txtlines_dict = page_file.get_article_dict()
    except():
        print("!! Can not load the lines of the Page XML {} !!\n".format(path_to_pagexml))
        return art_polygons_dict

    for article_id in art_txtlines_dict:
        for txtline in art_txtlines_dict[article_id]:
            try:
                # get the baseline of the text line as polygon
                polygon = txtline.baseline.to_polygon()
                # skipp baselines with less than two points
                if len(polygon.x_points) == len(polygon.y_points) > 1:
                    if article_id in art_polygons_dict:
                        art_polygons_dict[article_id].append(polygon)
                    else:
                        art_polygons_dict.update({article_id: [polygon]})
            except():
                print("!! 'NoneType' object with id {} has no attribute 'to_polygon' !!\n".format(txtline.id))
                continue

    return art_polygons_dict


def compute_baseline_detection_measure(polygon_dict_gt, polygon_dict_hy,
                                       min_tol=10, max_tol=30, rel_tol=0.25, poly_tick_dist=5):
    """
    :param polygon_dict_gt: ground truth article / block ID's with corresponding lists of polygons
    :param polygon_dict_hy: hypotheses article / block ID's with corresponding lists of polygons

    :param min_tol: MINIMUM distance tolerance which is not penalized
    :param max_tol: MAXIMUM distance tolerance which is not penalized
    :param rel_tol: fraction of estimated interline distance as tolerance values
    :param poly_tick_dist: desired distance (measured in pixels) of two adjacent pixels in the normed polygons

    :return: baseline detection measure ,i.e., r and p value (for all baselines and only for baselines assigned to
    articles / blocks)
    """
    list_of_gt_polygons, list_of_gt_polygons_without_none = [], []
    list_of_hy_polygons, list_of_hy_polygons_without_none = [], []

    for gt_article_id in polygon_dict_gt:
        list_of_gt_polygons += polygon_dict_gt[gt_article_id]
        if gt_article_id is not None:
            list_of_gt_polygons_without_none += polygon_dict_gt[gt_article_id]

    for hy_article_id in polygon_dict_hy:
        list_of_hy_polygons += polygon_dict_hy[hy_article_id]
        if hy_article_id is not None:
            list_of_hy_polygons_without_none += polygon_dict_hy[hy_article_id]

    print("{:<100s} {:>10d} {:<1s} {:>10d}".
          format("number of ground truth baselines / hypotheses baselines",
                 len(list_of_gt_polygons), "/", len(list_of_hy_polygons)))
    print("{:<100s} {:>10d} {:<1s} {:>10d}".
          format("number of ground truth baselines with article ID's / hypotheses baselines with article ID's",
                 len(list_of_gt_polygons_without_none), "/", len(list_of_hy_polygons_without_none)))

    # create baseline measure evaluation
    bl_measure_eval = \
        BaselineMeasureEval(min_tol=min_tol, max_tol=max_tol, rel_tol=rel_tol, poly_tick_dist=poly_tick_dist)

    # baseline detection measure for all baselines
    if len(list_of_gt_polygons) == 0:
        r_value_bd, p_value_bd = None, None
    elif len(list_of_hy_polygons) == 0:
        r_value_bd, p_value_bd = 0, 0
    else:
        bl_measure_eval.calc_measure_for_page_baseline_polys(polys_truth=list_of_gt_polygons,
                                                             polys_reco=list_of_hy_polygons)
        r_value_bd = bl_measure_eval.measure.result.page_wise_recall[-1]
        p_value_bd = bl_measure_eval.measure.result.page_wise_precision[-1]

    # baseline detection measure only for baselines assigned to articles / blocks
    if len(list_of_gt_polygons_without_none) == 0:
        r_value_bd_without_none, p_value_bd_without_none = None, None
    elif len(list_of_hy_polygons_without_none) == 0:
        r_value_bd_without_none, p_value_bd_without_none = 0, 0
    else:
        bl_measure_eval.calc_measure_for_page_baseline_polys(polys_truth=list_of_gt_polygons_without_none,
                                                             polys_reco=list_of_hy_polygons_without_none)
        r_value_bd_without_none = bl_measure_eval.measure.result.page_wise_recall[-1]
        p_value_bd_without_none = bl_measure_eval.measure.result.page_wise_precision[-1]

    return r_value_bd, p_value_bd, r_value_bd_without_none, p_value_bd_without_none


def get_greedy_sum(array):
    """
    :param array: matrix as numpy array

    :return: greedy sum of the given matrix
    """
    matrix = np.copy(array)
    s = 0

    while True:
        # calculate indices for maximum element
        max_idx_x, max_idx_y = np.unravel_index(np.argmax(matrix), matrix.shape)
        # finish if all elements have been considered
        if matrix[max_idx_x, max_idx_y] < 0:
            break

        # get max element
        s += matrix[(max_idx_x, max_idx_y)]
        # set row and column to -1
        matrix[max_idx_x, :] = -1.0
        matrix[:, max_idx_y] = -1.0

    return s


def run_eval(gt_file, hy_file, min_tol=10, max_tol=30, rel_tol=0.25, poly_tick_dist=5):
    """
    :param gt_file: ground truth Page XML file (with baselines and article / block ID's)
    :param hy_file: hypotheses Page XML file (with baselines and article / block ID's)

    :param min_tol: MINIMUM distance tolerance which is not penalized
    :param max_tol: MAXIMUM distance tolerance which is not penalized
    :param rel_tol: fraction of estimated interline distance as tolerance values
    :param poly_tick_dist: desired distance (measured in pixels) of two adjacent pixels in the normed polygons

    :return: baseline detection measure, baseline detection measure only for baselines assigned to articles / blocks and
    the article / block segmentation measure
    """
    if not gt_file.endswith(".xml") or not hy_file.endswith(".xml"):
        print("!! Ground truth and hypotheses file have to be in Page XML format !!\n")
        return None, None, None

    gt_polygons_dict = get_data_from_pagexml(path_to_pagexml=gt_file)
    hy_polygons_dict = get_data_from_pagexml(path_to_pagexml=hy_file)

    bd_r_value, bd_p_value, bd_r_value_without_none, bd_p_value_without_none \
        = compute_baseline_detection_measure(polygon_dict_gt=gt_polygons_dict, polygon_dict_hy=hy_polygons_dict,
                                             min_tol=min_tol, max_tol=max_tol, rel_tol=rel_tol,
                                             poly_tick_dist=poly_tick_dist)

    if bd_r_value is None:
        print("!! Ground truth Page XML has no baselines !!\n")
        return None, None, None
    if bd_r_value_without_none is None:
        print("!! Ground truth Page XML has no article / block ID's !!\n")
        bd_f_value = f_measure(recall=bd_r_value, precision=bd_p_value)
        return (bd_r_value, bd_p_value, bd_f_value), None, None

    bd_f_value = f_measure(recall=bd_r_value, precision=bd_p_value)
    bd_f_value_without_none = f_measure(recall=bd_r_value_without_none, precision=bd_p_value_without_none)

    # baselines without an article / block ID are irrelevant for our measure
    gt_polygons_dict.pop(None, None)
    # number of GT articles
    number_of_gt_articles = len(gt_polygons_dict)

    hy_polygons_dict.pop(None, None)
    # number of HY articles
    number_of_hy_articles = len(hy_polygons_dict)

    print("{:<100s} {:>10d} {:<1s} {:>10d}\n".
          format("number of ground truth articles / hypotheses articles",
                 number_of_gt_articles, "/", number_of_hy_articles))

    if number_of_hy_articles == 0:
        return (bd_r_value, bd_p_value, bd_f_value), \
               (bd_r_value_without_none, bd_p_value_without_none, bd_f_value_without_none), (0, 0, 0)

    ##########
    # computation of the weighted r and p matrix
    r_matrix = np.zeros((number_of_gt_articles, number_of_hy_articles), dtype=np.float)
    p_matrix = np.zeros((number_of_gt_articles, number_of_hy_articles), dtype=np.float)

    # create baseline measure evaluation
    bl_measure_eval = BaselineMeasureEval(min_tol=min_tol, max_tol=max_tol, rel_tol=rel_tol,
                                          poly_tick_dist=poly_tick_dist)

    hy_weighting_append = True
    gt_block_weighting_factors = []
    hy_block_weighting_factors = []

    # baseline detection measure between every ground truth and hypotheses article / block
    for gt_article_index, gt_article_id in enumerate(gt_polygons_dict):
        gt_block_weighting_factors.append(float(len(gt_polygons_dict[gt_article_id])))

        for hy_article_index, hy_article_id in enumerate(hy_polygons_dict):
            if hy_weighting_append:
                hy_block_weighting_factors.append(float(len(hy_polygons_dict[hy_article_id])))

            bl_measure_eval.calc_measure_for_page_baseline_polys(polys_truth=gt_polygons_dict[gt_article_id],
                                                                 polys_reco=hy_polygons_dict[hy_article_id])
            r_matrix[gt_article_index, hy_article_index] = bl_measure_eval.measure.result.page_wise_recall[-1]
            p_matrix[gt_article_index, hy_article_index] = bl_measure_eval.measure.result.page_wise_precision[-1]

        hy_weighting_append = False

    # multiplication of the rows (row-wise weighting for recall) / columns (column-wise weighting for precision)
    # by the corresponding weighting factors
    gt_block_weighting = \
        np.asarray([1 / sum(gt_block_weighting_factors) * x for x in gt_block_weighting_factors], dtype=np.float)
    hy_block_weighting = \
        np.asarray([1 / sum(hy_block_weighting_factors) * x for x in hy_block_weighting_factors], dtype=np.float)

    r_matrix = r_matrix * np.expand_dims(gt_block_weighting, axis=1)
    p_matrix = p_matrix * hy_block_weighting

    as_r_value = get_greedy_sum(array=r_matrix)
    as_p_value = get_greedy_sum(array=p_matrix)
    as_f_value = f_measure(recall=as_r_value, precision=as_p_value)

    return (bd_r_value, bd_p_value, bd_f_value), \
           (bd_r_value_without_none, bd_p_value_without_none, bd_f_value_without_none), \
           (as_r_value, as_p_value, as_f_value)


def run_measure(gt_files, hy_files, min_tol, max_tol, rel_tol, poly_tick_dist, verbose=True):
    if len(gt_files) != len(hy_files):
        print(f"Length of GT list ({len(gt_files)}) has to match length of HY list ({len(hy_files)})!")
        exit(1)

    # start java virtual machine to be able to execute the java code
    jpype.startJVM(jpype.getDefaultJVMPath())

    bd_average, bd_counter = [0, 0, 0], 0
    bd_without_none_average, bd_without_none_counter = [0, 0, 0], 0
    as_average, as_counter = [0, 0, 0], 0

    if verbose:
        for i, (gt_file, hy_file) in enumerate(zip(gt_files, hy_files)):
            print("-" * 125)
            print("Ground truth file: ", gt_file)
            print("Hypotheses file  : ", hy_file, "\n")

            tuple_bd, tuple_bd_without_none, tuple_as = run_eval(gt_file=gt_file, hy_file=hy_file,
                                                                 min_tol=min_tol, max_tol=max_tol,
                                                                 rel_tol=rel_tol, poly_tick_dist=poly_tick_dist)

            print("{:<50s} {:>10s} {:>10s} {:>10s}".format("Mode", "R-value", "P-value", "F-value"))

            if tuple_bd is not None:
                print("{:<50s} {:>10f} {:>10f} {:>10f}".
                      format("baseline detection measure - all baselines", tuple_bd[0], tuple_bd[1], tuple_bd[2]))

                bd_average = [bd_average[i] + tuple_bd[i] for i in range(len(bd_average))]
                bd_counter += 1
            else:
                print("{:<50s} {:>10s} {:>10s} {:>10s}".
                      format("baseline detection measure - all baselines", "-", "-", "-"))

            if tuple_bd_without_none is not None:
                print("{:<50s} {:>10f} {:>10f} {:>10f}".
                      format("baseline detection measure - without none",
                             tuple_bd_without_none[0], tuple_bd_without_none[1], tuple_bd_without_none[2]))

                bd_without_none_average = \
                    [bd_without_none_average[i] + tuple_bd_without_none[i] for i in range(len(bd_without_none_average))]
                bd_without_none_counter += 1
            else:
                print("{:<50s} {:>10s} {:>10s} {:>10s}".
                      format("baseline detection measure - without none", "-", "-", "-"))

            if tuple_as is not None:
                print("{:<50s} {:>10f} {:>10f} {:>10f}".
                      format("article / block segmentation measure", tuple_as[0], tuple_as[1], tuple_as[2]))

                as_average = [as_average[i] + tuple_as[i] for i in range(len(as_average))]
                as_counter += 1
            else:
                print("{:<50s} {:>10s} {:>10s} {:>10s}".
                      format("article / block segmentation measure", "-", "-", "-"))
    else:
        for i, (gt_file, hy_file) in enumerate(zip(gt_files, hy_files)):
            tuple_bd, tuple_bd_without_none, tuple_as = run_eval(gt_file=gt_file, hy_file=hy_file,
                                                                 min_tol=min_tol, max_tol=max_tol,
                                                                 rel_tol=rel_tol, poly_tick_dist=poly_tick_dist)

            if tuple_bd is not None:
                bd_average = [bd_average[i] + tuple_bd[i] for i in range(len(bd_average))]
                bd_counter += 1

            if tuple_bd_without_none is not None:
                bd_without_none_average = \
                    [bd_without_none_average[i] + tuple_bd_without_none[i] for i in range(len(bd_without_none_average))]
                bd_without_none_counter += 1

            if tuple_as is not None:
                as_average = [as_average[i] + tuple_as[i] for i in range(len(as_average))]
                as_counter += 1

    print("-" * 125)
    print("-" * 125)
    print("AVERAGE VALUES")
    print("{:<50s} {:>10s} {:>10s} {:>10s} {:>25s} {:>10s}".
          format("Mode", "R-value", "P-value", "F-value", "valid evaluated files", "all files"))

    if bd_counter > 0:
        print("{:<50s} {:>10f} {:>10f} {:>10f} {:>25d} {:>10d}".
              format("baseline detection measure - all baselines",
                     1 / bd_counter * bd_average[0], 1 / bd_counter * bd_average[1], 1 / bd_counter * bd_average[2],
                     bd_counter, len(gt_files)))
    else:
        print("{:<50s} {:>10s} {:>10s} {:>10s} {:>25d} {:>10d}".
              format("baseline detection measure - all baselines", "-", "-", "-", bd_counter, len(gt_files)))

    if bd_without_none_counter > 0:
        print("{:<50s} {:>10f} {:>10f} {:>10f} {:>25d} {:>10d}".
              format("baseline detection measure - without none",
                     1 / bd_without_none_counter * bd_without_none_average[0],
                     1 / bd_without_none_counter * bd_without_none_average[1],
                     1 / bd_without_none_counter * bd_without_none_average[2],
                     bd_without_none_counter, len(gt_files)))
    else:
        print("{:<50s} {:>10s} {:>10s} {:>10s} {:>25d} {:>10d}".
              format("baseline detection measure - without none", "-", "-", "-",
                     bd_without_none_counter, len(gt_files)))

    if as_counter > 0:
        print("{:<50s} {:>10f} {:>10f} {:>10f} {:>25d} {:>10d}".
              format("article / block segmentation measure",
                     1 / as_counter * as_average[0], 1 / as_counter * as_average[1], 1 / as_counter * as_average[2],
                     as_counter, len(gt_files)))
    else:
        print("{:<50s} {:>10s} {:>10s} {:>10s} {:>25d} {:>10d}".
              format("article / block segmentation measure", "-", "-", "-", as_counter, len(gt_files)))

    # shut down the java virtual machine
    jpype.shutdownJVM()


if __name__ == "__main__":
    parser = ArgumentParser()
    # command-line arguments
    parser.add_argument('--path_to_gt_xml_lst', type=str, required=True,
                        help="path to the lst file containing the file paths of the ground truth Page XML's")
    parser.add_argument('--path_to_hy_xml_lst', type=str, required=True,
                        help="path to the lst file containing the file paths of the hypotheses Page XML's")

    parser.add_argument('--min_tol', type=int, default=-1,
                        help="MINIMUM distance tolerance which is not penalized, -1 for dynamic calculation")
    parser.add_argument('--max_tol', type=int, default=-1,
                        help="MAXIMUM distance tolerance which is not penalized, -1 for dynamic calculation")
    parser.add_argument('--rel_tol', type=float, default=0.25,
                        help="fraction of estimated interline distance as tolerance values")
    parser.add_argument('--poly_tick_dist', type=int, default=5,
                        help="desired distance (measured in pixels) of two adjacent pixels in the normed polygons")
    parser.add_argument('--verbose', type=bool, default=True,
                        help="print evaluation for every single file in addition to overall summary")

    flags = parser.parse_args()

    # list of xml file paths
    gt_xml_files = [line.rstrip('\n') for line in open(flags.path_to_gt_xml_lst, "r")]
    hy_xml_files = [line.rstrip('\n') for line in open(flags.path_to_hy_xml_lst, "r")]
    # filter hy files by gt file (for train, val, test splits)
    gt_base_names = [os.path.splitext(os.path.basename(file))[0] for file in gt_xml_files]
    hy_xml_files = list(sorted([file for file in hy_xml_files if any([gt in os.path.basename(file) for gt in gt_base_names])], key=os.path.basename))
    gt_xml_files = list(sorted(gt_xml_files, key=os.path.basename))

    run_measure(gt_xml_files, hy_xml_files, flags.min_tol, flags.max_tol, flags.rel_tol,
                flags.poly_tick_dist, flags.verbose)
