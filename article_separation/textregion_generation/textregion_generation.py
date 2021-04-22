# -*- coding: utf-8 -*-

import jpype
import numpy as np
from argparse import ArgumentParser

from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.page_objects import TextRegion, Points

from citlab_python_util.geometry.util import alpha_shape
from citlab_python_util.geometry.polygon import norm_poly_dists

from citlab_article_separation.baseline_clustering.dbscan_baselines import get_list_of_interline_distances


def get_data_from_pagexml(path_to_pagexml, des_dist=50, max_d=500, use_java_code=True):
    """
    :param path_to_pagexml: file path
    :param des_dist: desired distance (measured in pixels) of two adjacent pixels in the normed polygons
    :param max_d: maximum distance (measured in pixels) for the calculation of the interline distances
    :param use_java_code: usage of methods written in java (faster than python!) or not

    :return: two dictionaries: {article id: corresponding list of text lines}
                               {text line id: (normed polygon, interline distance)}
    """
    # load the page xml file
    page_file = Page(path_to_pagexml)

    # get all text lines article wise
    art_txtlines_dict = page_file.get_article_dict()
    # get all text lines of the loaded page file
    lst_of_txtlines = page_file.get_textlines()

    lst_of_polygons = []
    lst_of_txtlines_adjusted = []

    for txtline in lst_of_txtlines:
        try:
            # get the baseline of the text line as polygon
            baseline = txtline.baseline.to_polygon()
            # baselines with less than two points will skipped
            if len(baseline.x_points) == len(baseline.y_points) > 1:
                lst_of_polygons.append(txtline.baseline.to_polygon())
                lst_of_txtlines_adjusted.append(txtline)
        except(AttributeError):
            # print("'NoneType' object in PAGEXML with id {} has no attribute 'to_polygon'!\n".format(txtline.id))
            continue

    # normed polygons
    lst_of_normed_polygons = norm_poly_dists(poly_list=lst_of_polygons, des_dist=des_dist)
    # interline distances
    lst_of_intdists = get_list_of_interline_distances(lst_of_polygons=lst_of_polygons, max_d=max_d,
                                                      use_java_code=use_java_code)

    txtline_dict = {}
    for i, txtline in enumerate(lst_of_txtlines_adjusted):
        # check the surrounding polygon of the text line
        if txtline.surr_p is None:
            normed_polygon = lst_of_normed_polygons[i]

            x_points_shifted = [x + 1 for x in normed_polygon.x_points]
            # y values are shifted upwards by at least one pixel
            y_shift = max(int(0.95 * lst_of_intdists[i]), 1)
            y_points_shifted = [y - y_shift for y in normed_polygon.y_points]

            sp_points = list(zip(normed_polygon.x_points + x_points_shifted[::-1],
                                 normed_polygon.y_points + y_points_shifted[::-1]))

            for article in art_txtlines_dict:
                for reference_txtline in art_txtlines_dict[article]:
                    if reference_txtline.id == txtline.id:
                        reference_txtline.surr_p = Points(sp_points)

        txtline_dict.update({txtline.id: (lst_of_normed_polygons[i], lst_of_intdists[i])})

    return art_txtlines_dict, txtline_dict


def txtlines_set_reading_order(lst_of_txtlines):
    """ Sorts a list of text lines regarding to the y coordinates and
        sets the corresponding reading order in the custom tag.

    :param lst_of_txtlines: list of baselines getting a reading order
    """
    y_centers_and_txtline = []

    for txtline in lst_of_txtlines:
        polygon = txtline.baseline.to_polygon()
        # y center value
        y_center = 1 / len(polygon.y_points) * sum(polygon.y_points)

        y_centers_and_txtline.append((y_center, txtline))

    y_centers_and_txtline.sort(key=lambda x: x[0])

    # set the reading order
    for reading_order, y in enumerate(y_centers_and_txtline):
        y[1].custom["readingOrder"] = {"index": reading_order}


def save_results_in_pagexml(path_to_pagexml, text_region_txtline_dict):
    """
    :param path_to_pagexml: file path
    :param text_region_txtline_dict: dictionray {text region id: (list of boundary_points,
                                                list of corresponding text lines, reading order of the region)}
    """
    page_file = Page(path_to_pagexml)
    lst_of_txtregions = []

    for txtregion_id in text_region_txtline_dict:
        boundary_points = text_region_txtline_dict[txtregion_id][0]
        lst_of_txtlines = text_region_txtline_dict[txtregion_id][1]
        reading_order = text_region_txtline_dict[txtregion_id][2]

        # set the reading order of the text lines
        txtlines_set_reading_order(lst_of_txtlines=lst_of_txtlines)

        # generation of the text region
        txtregion = TextRegion(_id=txtregion_id, region_type="paragraph",
                               custom={"readingOrder": {"index": reading_order}},
                               points=boundary_points, text_lines=lst_of_txtlines)
        lst_of_txtregions.append(txtregion)

    page_file.set_text_regions(text_regions=lst_of_txtregions, overwrite=True)
    page_file.write_page_xml(path_to_pagexml)


def create_text_regions(art_txtlines_dict, txtline_dict, alpha=75):
    """ Computation of boundary polygons of clustered baselines (,i.e., lines have an article tag!)
        based on the alpha shape algorithm.

    :param art_txtlines_dict: dictionary {article id: corresponding list of text lines}
    :param txtline_dict: dictionary {text line id: (normed polygon, interline distance)}
    :param alpha: alpha value for the alpha shape algorithm (for alpha -> infinity we get the convex hulls)

    :return: dictionray {text region id: (list of boundary_points, list of corresponding text lines,
                        reading order of the region)}
    """
    text_region_txtline_dict = {}
    text_region_counter = 0

    for article_id in art_txtlines_dict:
        if article_id is None:
            for txtline in art_txtlines_dict[article_id]:
                if txtline.id in txtline_dict:
                    normed_polygon = txtline_dict[txtline.id][0]

                    x_points_shifted = [x + 1 for x in normed_polygon.x_points]
                    # y values are shifted upwards by at least one pixel
                    y_shift = max(int(0.95 * txtline_dict[txtline.id][1]), 1)
                    y_points_shifted = [y - y_shift for y in normed_polygon.y_points]

                    np_points = list(zip(normed_polygon.x_points + x_points_shifted,
                                         normed_polygon.y_points + y_points_shifted))

                    # alpha shape boundary of the text line as integer points
                    boundary_points = alpha_shape(points=np.array(np_points), alpha=alpha)
                    boundary_points = [[int(j) for j in i] for i in boundary_points]

                    # reading order of the generated text region is simply "text_region_counter"
                    text_region_txtline_dict.update({"tr_" + str(text_region_counter):
                                                    (boundary_points, [txtline], text_region_counter)})
                    text_region_counter += 1
        else:
            np_points, lst_of_txtlines = [], []

            for txtline in art_txtlines_dict[article_id]:
                if txtline.id in txtline_dict:
                    lst_of_txtlines.append(txtline)
                    normed_polygon = txtline_dict[txtline.id][0]

                    x_points_shifted = [x + 1 for x in normed_polygon.x_points]
                    # y values are shifted upwards by at least one pixel
                    y_shift = max(int(0.95 * txtline_dict[txtline.id][1]), 1)
                    y_points_shifted = [y - y_shift for y in normed_polygon.y_points]

                    np_points += list(zip(normed_polygon.x_points + x_points_shifted,
                                          normed_polygon.y_points + y_points_shifted))

            # alpha shape boundary of the text lines as integer points
            boundary_points = alpha_shape(points=np.array(np_points), alpha=alpha)
            boundary_points = [[int(j) for j in i] for i in boundary_points]

            # reading order of the generated text region is simply "text_region_counter"
            text_region_txtline_dict.update({"tr_" + str(text_region_counter):
                                            (boundary_points, lst_of_txtlines, text_region_counter)})
            text_region_counter += 1

    return text_region_txtline_dict


if __name__ == "__main__":
    parser = ArgumentParser()
    # command-line arguments
    parser.add_argument('--path_to_xml_file', type=str, required=True,
                        help="path to the page xml file to be processed")

    parser.add_argument('--des_dist', type=int, default=50,
                        help="desired distance (measured in pixels) of two adjacent pixels in the normed polygons")
    parser.add_argument('--max_d', type=int, default=100,
                        help="maximum distance (measured in pixels) for the calculation of the interline distances")
    parser.add_argument('--alpha', type=float, default=75,
                        help="alpha value for the alpha shape algorithm "
                             "(for alpha -> infinity we get the convex hulls, recommended: alpha >= des_dist)")
    parser.add_argument('--use_java_code', type=bool, default=True,
                        help="usage of methods written in java (faster than python!) or not")

    flags = parser.parse_args()

    # start java virtual machine to be able to execute the java code
    jpype.startJVM(jpype.getDefaultJVMPath())

    xml_file = flags.path_to_xml_file
    print(xml_file)
    article_textlines_dict, textline_dict = \
        get_data_from_pagexml(path_to_pagexml=xml_file, des_dist=flags.des_dist, max_d=flags.max_d,
                              use_java_code=flags.use_java_code)

    text_region_textline_dict = \
        create_text_regions(art_txtlines_dict=article_textlines_dict, txtline_dict=textline_dict, alpha=flags.alpha)

    # TODO: except?
    save_results_in_pagexml(path_to_pagexml=xml_file, text_region_txtline_dict=text_region_textline_dict)

    # shut down the java virtual machine
    jpype.shutdownJVM()
