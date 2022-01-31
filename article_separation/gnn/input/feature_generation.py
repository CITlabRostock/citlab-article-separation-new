import os
import json
import re
import time
import logging
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from shapely.geometry import LineString
from article_separation.gnn.input.textblock_similarity import TextblockSimilarity
from python_util.image_processing.swt_dist_trafo import StrokeWidthDistanceTransform
from python_util.geometry.util import convex_hull, bounding_box
from python_util.io.path_util import get_img_from_page_path
from python_util.parser.xml.page.page import Page
from python_util.math.rounding import round_by_precision_and_base as round_base


def get_text_region_geometric_features(text_region, norm_x, norm_y):
    """
    Generates 4d geometric features for `text_region`.

    A 2d feature describing the size of the text region, i.e. a vector (w,h) spanning the `text_region` and
    a 2d feature describing the center (x,y) of the `text_region`.

    Each component of the two 2d features is normed by `norm_x` and `norm_y` respectively.

    :param text_region: TextRegion object
    :param norm_x: norm scalar for the width/x (usually image width)
    :param norm_y: norm scalar for the height/y (usually image height)
    :return: 4d geometric feature vector
    """
    tr_points = np.asarray(text_region.points.points_list, dtype=np.int32)
    # bounding box of text region
    min_x, max_x, min_y, max_y = get_bounding_box(tr_points)
    width = float(max_x) - float(min_x)
    height = float(max_y) - float(min_y)
    # feature vector describing the extension of the text region
    size_x = width / norm_x
    size_y = height / norm_y
    # feature vector describing the center of the text region
    center_x = (min_x + max_x) / (2 * norm_x)
    center_y = (min_y + max_y) / (2 * norm_y)
    # 4-dimensional feature
    return [size_x, size_y, center_x, center_y]


def get_text_region_baseline_features(text_region, norm_x, norm_y):
    """
    Generates 8d baseline features for `text_region`.

    Picks the top and bottom baseline of the text region and computes 4d geometric features for each, describing
    their size and center.

    Features regarding the width or x-coordinate are normed by `norm_x`, whereas features regarding the height
    or y-coordinate are normed by `norm_y`.

    :param text_region: TextRegion object
    :param norm_x: norm scalar for the width/x (usually image width)
    :param norm_y: norm scalar for the height/y (usually image height)
    :return: 8d baseline feature vector
    """
    feature = []
    # geometric information about top & bottom textline of text region
    top_baseline = text_region.text_lines[0].baseline
    bottom_baseline = text_region.text_lines[-1].baseline
    for baseline in (top_baseline, bottom_baseline):
        # bounding box of baseline
        points_baseline = np.asarray(baseline.points_list, dtype=np.int32)
        min_x, max_x, min_y, max_y = get_bounding_box(points_baseline)
        width = float(max_x) - float(min_x)
        height = float(max_y) - float(min_y)
        # feature vector describing the extension of the baseline
        size_x = width / norm_x
        size_y = height / norm_y
        # feature vector describing the center of the baseline
        center_x = (min_x + max_x) / (2 * norm_x)
        center_y = (min_y + max_y) / (2 * norm_y)
        # extend feature
        feature.extend([size_x, size_y, center_x, center_y])
    # return 8-dimensional feature
    return feature


def get_text_regions_wv_sim(text_regions, feature_extractor):
    """
    Generates a feature dictionary with entries for every possible pair of text regions in `text_regions`. The entries
    include text block similarity scores based on language-specific pretrained word vectors, where the full text
    embeddings of each text region are compared via a cosine-like similarity.

    :param text_regions: list of TextRegion objects
    :param feature_extractor: TextblockSimilarity object
    :return: feature dictionary
    """
    # build {tb : text} dict
    tb_dict = dict()
    for text_region in text_regions:
        text = "\n".join([text_line.text for text_line in text_region.text_lines])
        tb_dict[text_region.id] = text
    # run feature extractor
    feature_extractor.set_tb_dict(tb_dict)
    feature_extractor.run()
    return feature_extractor.feature_dict


def get_textline_stroke_widths_heights_dist_trafo(page_path, text_lines, img_path=None):
    """
    Generates two feature dictionaries with entries for each text line given in `text_lines`. The first
    feature contains an approximate stroke width of the text line. The second feature contains an approximate
    text height of the text line.

    A Distance Transform is used on the image file corresponding to the pageXML file given in `page_path`.
    In this transformed image, connected components over the text line bounding boxes are analysed to compute the
    features.

    The stroke width of a text line is set as the median value over the maximum distance values of all contained
    connected components.

    The text height of a text line is set as the maximum height over all ccontained onnected components.

    Prior to this computation, connected components with unreasonable size or aspect ratio get discarded.

    :param page_path: path to pageXML file
    :param text_lines: list of TextLine objects in the pageXML file
    :param img_path: (optional) path to image file to compute the Distance Transform
    :return: (dict1, dict2) where `dict2` contains stroke width features and `dict2` contains text height features
    """
    if img_path is None:
        img_path = get_img_from_page_path(page_path)
    if not img_path:
        raise ValueError(f"Could not find corresponding image file to pagexml '{page_path}'")
    # run SWT
    SWT = StrokeWidthDistanceTransform(dark_on_bright=True)
    swt_img = SWT.distance_transform(img_path)
    # compute stroke widths and text heights on text line level
    textline_stroke_widths = dict()
    textline_heights = dict()
    for text_line in text_lines:
        # build surrounding polygons over text lines
        points_text_line = np.asarray(text_line.surr_p.points_list, dtype=np.int32)
        min_x, max_x, min_y, max_y = get_bounding_box(points_text_line)
        # get swt for text line
        text_line_swt = swt_img[min_y:max_y + 1, min_x:max_x + 1]
        # get connected components in text line
        text_line_ccs = SWT.connected_components_cv(text_line_swt)
        # remove CCs with unreasonable size or aspect ratio
        text_line_ccs = SWT.clean_connected_components(text_line_ccs)
        # go over connected components to estimate stroke width and text height of the text line
        swt_cc_values = []
        text_line_height = 0
        for cc in text_line_ccs:
            # component is a 4-tuple (x, y, width, height)
            # take max value in distance_transform as stroke_width for current CC (can be 0)
            swt_cc_values.append(np.max(text_line_swt[cc[1]: cc[1] + cc[3], cc[0]: cc[0] + cc[2]]))
            # new text height
            if cc[3] > text_line_height:
                text_line_height = cc[3]
        textline_stroke_widths[text_line.id] = np.median(swt_cc_values) if swt_cc_values else 0.0
        textline_heights[text_line.id] = text_line_height
    return textline_stroke_widths, textline_heights


def get_text_region_stroke_width_feature(text_region, textline_stroke_widths, norm=1.0):
    """
    Generates 1d stroke width feature for `text_region`.

    Takes a feature dictionary `textline_stroke_widths`, extracts the text line features corresponding to the given
    text region and then computes the maximum value over those features with an optional normalization factor.

    :param text_region: TextRegion object
    :param textline_stroke_widths: dictionary containing stroke width features for text lines
    :param norm: (optional) normalization factor for resulting feature
    :return: 1d stroke width feature
    """
    # 0-feature for empty text regions
    if all([not line.text for line in text_region.text_lines]):
        return [0.0]
    # maximum stroke width over text lines of text region
    # we prefer the maximum, so headings that are clustered in a block with other text dont get averaged out
    else:
        text_region_stroke_widths = [textline_stroke_widths[line.id] for line in text_region.text_lines if line.text]
        text_region_stroke_width = np.max(text_region_stroke_widths) / norm
        return [text_region_stroke_width]


def get_text_region_text_height_feature(text_region, textline_heights, norm=1.0):
    """
    Generates 1d text height feature for `text_region`.

    Takes a feature dictionary `textline_heights`, extracts the text line features corresponding to the given
    text region and then computes the maximum value over those features with an optional normalization factor.

    :param text_region: TextRegion object
    :param textline_heights: dictionary containing text height features for text lines
    :param norm: (optional) normalization factor for resulting feature
    :return: 1d text height feature
    """
    # 0-feature for empty text regions
    if all([not line.text for line in text_region.text_lines]):
        return [0.0]
    # maximum text height over text lines of text region
    # we prefer the maximum, so headings that are clustered in a block with other text dont get averaged out
    else:
        text_region_line_heights = [textline_heights[line.id] for line in text_region.text_lines if line.text]
        text_region_text_height = np.max(text_region_line_heights) / norm
        return [text_region_text_height]


def get_text_region_heading_feature(text_region):
    """
    Generates 1d (binary) heading feature for `text_region`.

    Checks whether or not the region type is a 'heading' and computes a corresponding binary feature.

    :param text_region: TextRegion object
    :return: 1d (binary) heading feature
    """
    contains_heading = True if text_region.region_type.lower() == 'heading' else False
    return [float(contains_heading)]


def get_edge_separator_feature_line(text_region_a, text_region_b, separator_regions):
    """
    Generates 2d (binary) separator feature for a pair text regions.

    Checks if `text_region_a` and `text_region_b` are separated by SeparatorRegions given in `separator_regions`.
    It differentiates between horizontal and vertical separators and computes a separate feature for both.
    It is based on edge intersections with separator regions.

    :param text_region_a: TextRegion object
    :param text_region_b: TextRegion object
    :param separator_regions: list of SeparatorRegion objects
    :return: 2d (binary) separator feature (horizontal, vertical)
    """
    # surrounding polygons of text regions
    points_a = np.asarray(text_region_a.points.points_list, dtype=np.int32)
    points_b = np.asarray(text_region_b.points.points_list, dtype=np.int32)
    # bounding boxes of text regions
    min_x_a, max_x_a, min_y_a, max_y_a = get_bounding_box(points_a)
    min_x_b, max_x_b, min_y_b, max_y_b = get_bounding_box(points_b)
    # center of text regions
    center_x_a = (min_x_a + max_x_a) / 2
    center_y_a = (min_y_a + max_y_a) / 2
    center_x_b = (min_x_b + max_x_b) / 2
    center_y_b = (min_y_b + max_y_b) / 2
    # visual line connecting both text regions
    tr_segment = LineString([(center_x_a, center_y_a), (center_x_b, center_y_b)])
    # go over seperator regions and check for intersections
    horizontally_separated = False
    vertically_separated = False
    for separator_region in separator_regions:
        # surrounding polygon of separator region
        points_s = np.asarray(separator_region.points.points_list, dtype=np.int32)
        # bounding box of separator region
        min_x_s, max_x_s, min_y_s, max_y_s = get_bounding_box(points_s)
        # height-width-ratio of bounding box
        width = max(max_x_s - min_x_s, 1)
        height = max(max_y_s - min_y_s, 1)
        ratio = float(height) / float(width)
        # corner points of bounding box
        s1 = (min_x_s, min_y_s)
        s2 = (max_x_s, min_y_s)
        s3 = (min_x_s, max_y_s)
        s4 = (max_x_s, max_y_s)
        # check for intersections/containment between tr_segment and bounding box as prior test
        if line_poly_intersection(tr_segment, [s1, s2, s3, s4]) or \
                line_in_bounding_box(tr_segment, min_x_s, max_x_s, min_y_s, max_y_s):
            # check for intersections between text_region_line and surrounding polygon
            if line_poly_intersection(tr_segment, separator_region.points.points_list):
                sep_orientation = separator_region.get_orientation()
                if sep_orientation == 'horizontal':
                    horizontally_separated = True
                elif separator_region == 'vertical':
                    vertically_separated = True
                # ratio check
                else:
                    logging.debug(f"No custom orientation tag found for separator region. Defaulting to ratio check.")
                    if ratio < 5:
                        horizontally_separated = \
                            True
                    else:
                        vertically_separated = True
                if horizontally_separated and vertically_separated:
                    break
    separator_feature = [float(horizontally_separated), float(vertically_separated)]
    logging.debug(f"{text_region_a.id} - {text_region_b.id}: {separator_feature}")
    return separator_feature


def get_bounding_box(points):
    """Returns the bounding box over a set of points."""
    min_x_a, max_x_a = np.min(points[:, 0]), np.max(points[:, 0])
    min_y_a, max_y_a = np.min(points[:, 1]), np.max(points[:, 1])
    return min_x_a, max_x_a, min_y_a, max_y_a


def line_poly_intersection(line, polygon):
    """Checks if LineString `line` intersects `polygon` (list of 2d points)."""
    # Optionally close polygon
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    # Go over polygon segments and check for intersection with line
    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        segment = LineString([p1, p2])
        if line.intersects(segment):
            return True
    return False


def line_in_bounding_box(line, min_x, max_x, min_y, max_y):
    """Checks if LineString `line` is contained in bounding box given by `min_x`, `max_x`, `min_y`, `max_y`."""
    x1, y1, x2, y2 = line.bounds
    if x1 > min_x and x2 < max_x and y1 > min_y and y2 < max_y:
        return True
    return False


def get_edge_separator_feature_bb(text_region_a, text_region_b, separator_regions):
    """
    Generates 2d (binary) separator feature for a pair text regions.

    Checks if `text_region_a` and `text_region_b` are separated by SeparatorRegions given in `separator_regions`.
    It differentiates between horizontal and vertical separators and computes a separate feature for both.
    It is based on rules regarding the bounding boxes of the regions.

    :param text_region_a: TextRegion object
    :param text_region_b: TextRegion object
    :param separator_regions: list of SeparatorRegion objects
    :return: 2d (binary) separator feature (horizontal, vertical)
    """
    # surrounding polygons of text regions
    points_a = np.asarray(text_region_a.points.points_list, dtype=np.int32)
    points_b = np.asarray(text_region_b.points.points_list, dtype=np.int32)
    # bounding boxes of text regions
    bb_a = get_bounding_box(points_a)
    bb_b = get_bounding_box(points_b)
    # go over seperator regions and check for rules
    horizontally_separated = False
    vertically_separated = False
    for separator_region in separator_regions:
        # surrounding polygon of separator region
        points_sep = np.asarray(separator_region.points.points_list, dtype=np.int32)
        # bounding box of separator region
        bb_sep = get_bounding_box(points_sep)
        # separator orientation
        orientation = separator_region.get_orientation()
        if orientation is None:
            # ratio check of bounding box
            width = max(bb_sep[1] - bb_sep[0], 1)
            height = max(bb_sep[3] - bb_sep[2], 1)
            ratio = float(height) / float(width)
            orientation = "horizontal" if ratio < 5 else "vertical"
        # rule checks
        if orientation == "vertical":
            if is_vertically_separated(*bb_a, *bb_b, *bb_sep):
                vertically_separated = True
        else:
            if is_horizontally_separated(*bb_a, *bb_b, *bb_sep):
                horizontally_separated = True
        if horizontally_separated and vertically_separated:
            break
    separator_feature = [float(horizontally_separated), float(vertically_separated)]
    logging.debug(f"{text_region_a.id} - {text_region_b.id}: {separator_feature}")
    return separator_feature


def is_vertically_separated(min_x_a, max_x_a, min_y_a, max_y_a,
                            min_x_b, max_x_b, min_y_b, max_y_b,
                            min_x_sep, max_x_sep, min_y_sep, max_y_sep):
    """Rule-based vertical separation criterion based on bounding boxes"""
    mean_x_sep = (min_x_sep + max_x_sep) / 2
    # not horizontally aligned
    if not ((max_x_a <= mean_x_sep <= min_x_b) or  # A - S - B
            (max_x_b <= mean_x_sep <= min_x_a)):  # B - S - A
        return False
    # not atleast one vertically aligned
    if not ((max_y_a >= min_y_sep and min_y_a <= max_y_sep) or  # A | S
            (max_y_b >= min_y_sep and min_y_b <= max_y_sep)):  # S | B
        return False
    return True


def is_horizontally_separated(min_x_a, max_x_a, min_y_a, max_y_a,
                              min_x_b, max_x_b, min_y_b, max_y_b,
                              min_x_sep, max_x_sep, min_y_sep, max_y_sep):
    """Rule-based horizontal separation criterion based on bounding boxes"""
    mean_y_sep = (min_y_sep + max_y_sep) / 2
    # not vertically aligned
    if not ((min_y_a <= mean_y_sep <= max_y_b) or  # A over S over B
            (min_y_b <= mean_y_sep <= max_y_a)):  # B over S over A
        return False
    # vertically aligned, but
    # both A & B outside of S on the same side
    if ((max_x_a <= min_x_sep and max_x_b <= min_x_sep) or  # both outside to the left
            (min_x_a >= max_x_sep and min_x_b >= max_x_sep)):  # both outside to the right
        return False
    return True


def is_aligned_horizontally_separated(text_region_a, text_region_b, separator_regions):
    """Function that determines whether two text regions are horizontally separated by a horizontal separator region,
    under the condition that they are vertically aligned"""
    # surrounding polygons of text regions
    points_a = np.asarray(text_region_a.points.points_list, dtype=np.int32)
    points_b = np.asarray(text_region_b.points.points_list, dtype=np.int32)
    # bounding boxes of text regions
    min_x_a, max_x_a, min_y_a, max_y_a = get_bounding_box(points_a)
    min_x_b, max_x_b, min_y_b, max_y_b = get_bounding_box(points_b)
    # go over seperator regions and check for rules
    for separator_region in separator_regions:
        # surrounding polygon of separator region
        points_s = np.asarray(separator_region.points.points_list, dtype=np.int32)
        # bounding box of separator region
        min_x_s, max_x_s, min_y_s, max_y_s = get_bounding_box(points_s)
        # separator orientation
        orientation = separator_region.get_orientation()
        if orientation is None:
            # ratio check of bounding box
            width = max(max_x_s - min_x_s, 1)
            height = max(max_y_s - min_y_s, 1)
            ratio = float(height) / float(width)
            orientation = "horizontal" if ratio < 5 else "vertical"
        # we only care about horizontal separators
        if orientation == 'vertical':
            continue
        # rule check
        # not vertically aligned
        mean_y_sep = (min_y_s + max_y_s) / 2
        if not ((min_y_a <= mean_y_sep <= max_y_b) or  # A over S over B
                (min_y_b <= mean_y_sep <= max_y_a)):  # B over S over A
            continue
        # not horizontally aligned
        if not ((max_x_a >= min_x_s and max_x_b >= min_x_s) and  # max offset to the left
                (min_x_a <= max_x_s and min_x_b <= max_x_s)):  # max offset to the right
            continue
        # is horizontally separated (and vertically aligned)
        return True


def is_aligned_heading_separated(text_region_a, text_region_b):
    # headings
    heading_a = text_region_a.region_type.lower() == 'heading'
    heading_b = text_region_b.region_type.lower() == 'heading'
    # both headings present
    if heading_a and heading_b:
        return False
    # no heading present
    if not (heading_a or heading_b):
        return False
    # surrounding polygons of text regions
    points_a = np.asarray(text_region_a.points.points_list, dtype=np.int32)
    points_b = np.asarray(text_region_b.points.points_list, dtype=np.int32)
    # bounding boxes of text regions
    min_x_a, max_x_a, min_y_a, max_y_a = get_bounding_box(points_a)
    min_x_b, max_x_b, min_y_b, max_y_b = get_bounding_box(points_b)
    # one heading present
    # not horizontally aligned
    if not (min_x_a <= max_x_b and min_x_b <= max_x_a):
        return False
    # one heading present
    if heading_a:
        # heading not vertically lower
        if not (min_y_a >= max_y_b):
            return False
    if heading_b:
        # heading not vertically lower
        if not (min_y_b >= max_y_a):
            return False
    # is heading separated
    return True


def get_node_visual_region(text_region):
    """Generates visual region for `text_region` as its bounding box."""
    # surrounding polygon of text region
    points = text_region.points.points_list
    # bounding box
    bb = bounding_box(points)
    return bb


def get_edge_visual_region(text_region_a, text_region_b):
    """Generates visual region regarding a pair of text regions as their convex hull."""
    # surrounding polygons of text regions
    points_a = text_region_a.points.points_list
    points_b = text_region_b.points.points_list
    # convex hull over both regions
    # TODO: Alternative to convex hull?!
    hull = convex_hull(points_a + points_b)
    return hull


def fully_connected_edges(num_nodes):
    """
    Generates a fully-connected edge set (excluding self-loops) for a graph with `num_nodes` nodes.

    :param num_nodes: number of nodes in the graph
    :return: 2d numpy-array representing the edge set
    """
    node_indices = np.arange(num_nodes, dtype=np.int32)
    node_indices = np.tile(node_indices, [num_nodes, 1])
    node_indices_t = np.transpose(node_indices)
    # fully-connected
    interacting_nodes = np.stack([node_indices_t, node_indices], axis=2).reshape([-1, 2])
    # remove self-loops
    del_indices = np.arange(num_nodes) * (num_nodes + 1)
    interacting_nodes = np.delete(interacting_nodes, del_indices, axis=0)
    return interacting_nodes


def delaunay_edges(num_nodes, node_positions):
    """
    Generates a Delaunay triangulation as the edge set for a graph with `num_nodes` nodes.

    :param num_nodes: number of nodes in the graph
    :param node_positions: 2d array containing the geometric positions of the nodes
    :return: 2d numpy-array representing the edge set
    """
    # round to nearest 50px for a more homogenous layout
    node_positions_smooth = round_base(node_positions, base=50)
    # interacting nodes are neighbours in the delaunay triangulation
    try:
        delaunay = Delaunay(node_positions_smooth)
    except QhullError:
        logging.warning("Delaunay input has the same x-coords. Defaulting to unsmoothed data.")
        delaunay = Delaunay(node_positions)
    indice_pointer, indices = delaunay.vertex_neighbor_vertices
    interacting_nodes = []
    for v in range(num_nodes):
        neighbors = indices[indice_pointer[v]:indice_pointer[v + 1]]
        interaction = np.stack(np.broadcast_arrays(v, neighbors), axis=1)
        interacting_nodes.append(interaction)
    interacting_nodes = np.concatenate(interacting_nodes, axis=0)
    return interacting_nodes


def get_data_from_pagexml(path_to_pagexml):
    """ Extracts information contained by in a pageXML file given by `path_to_pagexml`.

    :param path_to_pagexml: file path of the pageXML
    :return: dict of regions, list of text lines, list of baselines, list of article ids, image resolution
    """
    # load the page xml file
    page_file = Page(path_to_pagexml)

    # get text regions
    dict_of_regions = page_file.get_regions()

    # get all text lines
    list_of_txt_lines = page_file.get_textlines()
    list_of_baselines = []
    list_of_article_ids = []
    for txt_line in list_of_txt_lines:
        # get the baseline of the text line as polygon
        list_of_baselines.append(txt_line.baseline.to_polygon())
        # get the article id of the text line
        list_of_article_ids.append(txt_line.get_article_id())

    # image resolution
    resolution = page_file.get_image_resolution()
    return dict_of_regions, list_of_txt_lines, list_of_baselines, list_of_article_ids, resolution


def discard_text_regions_and_lines(text_regions, text_lines=None):
    # discard regions
    discard = 0
    text_lines_to_remove = []
    for tr in text_regions.copy():
        # ... without text lines
        if not tr.text_lines:
            text_regions.remove(tr)
            logging.debug(f"Discarding TextRegion {tr.id} (no textlines)")
            discard += 1
            continue
        # ... too small
        bounding_box = tr.points.to_polygon().get_bounding_box()
        if bounding_box.width < 10 or bounding_box.height < 10:
            text_regions.remove(tr)
            logging.debug(f"Discarding TextRegion {tr.id} (bounding box too small, height={bounding_box.height}, "
                          f"width={bounding_box.width})")
            if text_lines:
                for text_line in tr.text_lines:
                    text_lines_to_remove.append(text_line.id)
            discard += 1
    # discard corresponding text lines
    if text_lines_to_remove:
        text_lines = [line for line in text_lines if line.id not in text_lines_to_remove]
    if discard > 0:
        logging.warning(f"Discarded {discard} degenerate text_region(s). Either no text lines or region too small.")
    return text_regions, text_lines


def build_input_and_target(page_path,
                           interaction='delaunay',
                           visual_regions=False,
                           external_data=None,
                           sim_feat_extractor=None,
                           separators='bb'):
    """
    Computation of the input and target values to solve the article separation problem with a graph neural
    network on text region (baseline clusters) level.

    Generates the underlying graph structure (edge set), the node and edge features as well as the target
    ground truth relations.

    :param page_path: path to pageXML file
    :param interaction: method for edge set generation ('delaunay' or 'fully')
    :param visual_regions: (bool) optionally build visual regions for nodes and edges (default False)
    :param external_data: (optional) list of additonal feature dictionaries from external json sources
    :param sim_feat_extractor: (optional) TextblockSimilarity feature extractor
    :param separators: method for edge separator features ('bb' or 'line')
    :return: 'num_nodes', 'interacting_nodes', 'num_interacting_nodes' ,'node_features', 'edge_features',
        'visual_region_nodes', 'num_points_visual_region_nodes', 'visual_region_edges',
        'num_points_visual_region_edges', 'gt_relations', 'gt_num_relations'
    """
    assert interaction in ('fully', 'delaunay'), \
        f"Interaction setup {interaction} is not supported. Choose from ('fully', 'delaunay') instead."

    # load page data
    regions, text_lines, baselines, article_ids, resolution = get_data_from_pagexml(page_path)
    norm_x, norm_y = float(resolution[0]), float(resolution[1])
    try:
        text_regions = regions['TextRegion']
    except KeyError:
        logging.warning(f'No TextRegions found in {page_path}. Returning None.')
        return None, None, None, None, None, None, None, None, None, None, None

    # number of nodes
    num_nodes = len(text_regions)
    if num_nodes <= 1:
        logging.warning(f'Less than two nodes found in {page_path}. Returning None.')
        return None, None, None, None, None, None, None, None, None, None, None

    # pre-compute stroke width and height over textlines (and their maximum value for normalization)
    textline_stroke_widths, textline_heights = get_textline_stroke_widths_heights_dist_trafo(page_path, text_lines)
    sw_max = np.max(list(textline_stroke_widths.values()))
    th_max = np.max(list(textline_heights.values()))

    # node features
    node_features = []
    # compute region features
    for text_region in text_regions:
        node_feature = []
        # region geometric feature (4-dim)
        node_feature.extend(get_text_region_geometric_features(text_region, norm_x, norm_y))
        # top/bottom baseline geometric feature (8-dim)
        node_feature.extend(get_text_region_baseline_features(text_region, norm_x, norm_y))
        # stroke width feature (1-dim)
        node_feature.extend(get_text_region_stroke_width_feature(text_region, textline_stroke_widths, norm=sw_max))
        # text height feature (1-dim)
        node_feature.extend(get_text_region_text_height_feature(text_region, textline_heights, norm=th_max))
        # heading feature (1-dim)
        node_feature.extend(get_text_region_heading_feature(text_region))
        # external features
        if external_data:
            for ext in external_data:
                try:
                    ext_page = ext[os.path.basename(page_path)]
                except KeyError:
                    logging.warning(f'Could not find key {os.path.basename(page_path)} in external data json.')
                    continue
                if 'node_features' in ext_page:
                    try:
                        node_feature.extend(ext_page['node_features'][text_region.id])
                    except KeyError:
                        logging.debug(f"Could not find entry node_features->{text_region.id} in external json. "
                                      f"Defaulting.")
                        try:
                            node_feature.extend([ext_page['node_features']['default']])
                        except KeyError:
                            logging.debug(f"Could not find entry node_features->default in external json. Using 0.0.")
                            node_feature.extend([0.0])
        # final node feature vector
        node_features.append(node_feature)

    # interacting nodes (edge set)
    if interaction == 'fully' or num_nodes < 4:
        interacting_nodes = fully_connected_edges(num_nodes)
    else:  # delaunay
        node_centers = np.array(node_features, dtype=np.float32)[:, 2:4] * [norm_x, norm_y]
        interacting_nodes = delaunay_edges(num_nodes, node_centers)

    # number of interacting nodes
    num_interacting_nodes = interacting_nodes.shape[0]

    # pre-compute text block similarities with word vectors
    tb_sim_dict = get_text_regions_wv_sim(text_regions, sim_feat_extractor) if sim_feat_extractor is not None else None

    # regions for separator features
    separator_regions = regions['SeparatorRegion'] if 'SeparatorRegion' in regions else None

    # edge features for each pair of interacting nodes
    edge_features = []
    for i in range(num_interacting_nodes):
        edge_feature = []
        node_a, node_b = interacting_nodes[i, 0], interacting_nodes[i, 1]
        text_region_a, text_region_b = text_regions[node_a], text_regions[node_b]
        # separator feature (2-dim)
        if separator_regions:
            if separators == 'line':
                edge_feature.extend(get_edge_separator_feature_line(text_region_a, text_region_b, separator_regions))
            else:  # separators 'bb' default
                edge_feature.extend(get_edge_separator_feature_bb(text_region_a, text_region_b, separator_regions))
        else:
            edge_feature.extend([0.0, 0.0])
        # text block similarity features based on word vectors
        if tb_sim_dict:
            try:
                edge_feature.extend(tb_sim_dict['edge_features'][text_region_a.id][text_region_b.id])
            except KeyError:
                logging.debug(f"Could not find entry edge_features->{text_region_a.id}->{text_region_b.id} in "
                              f"text block similarity dict. Defaulting.")
                try:
                    edge_feature.extend(tb_sim_dict['edge_features']['default'])
                except KeyError:
                    logging.debug(f"Could not find entry edge_features->default in "
                                  f"text block similarity dict. Using 0.5.")
                    edge_feature.extend([0.5])
        # external features
        for ext in external_data:
            try:
                ext_page = ext[os.path.basename(page_path)]
            except KeyError:
                logging.warning(f'Could not find key {os.path.basename(page_path)} in external data json. Skipping.')
                continue
            if 'edge_features' in ext_page:
                try:
                    edge_feature.extend(ext_page['edge_features'][text_region_a.id][text_region_b.id])
                except (KeyError, TypeError):
                    logging.debug(f"Could not find entry edge_features->{text_region_a.id}->{text_region_b.id} in "
                                  f"external json. Defaulting.")
                    try:
                        edge_feature.extend(ext_page['edge_features']['default'])
                    except KeyError:
                        logging.debug(f"Could not find entry edge_features->default in external json. Using 0.5.")
                        edge_feature.extend([0.5])
        # final edge feature vector
        edge_features.append(edge_feature)

    # visual regions for nodes (for GNN visual features)
    visual_regions_nodes = []
    num_points_visual_regions_nodes = []
    if visual_regions:
        for text_region in text_regions:
            visual_region_node = get_node_visual_region(text_region)
            visual_regions_nodes.append(visual_region_node)
            num_points_visual_regions_nodes.append(len(visual_region_node))

    # visual regions for edges (for GNN visual features)
    visual_regions_edges = []
    num_points_visual_regions_edges = []
    if visual_regions:
        for i in range(num_interacting_nodes):
            node_a, node_b = interacting_nodes[i, 0], interacting_nodes[i, 1]
            text_region_a, text_region_b = text_regions[node_a], text_regions[node_b]
            visual_region_edge = get_edge_visual_region(text_region_a, text_region_b)
            visual_regions_edges.append(visual_region_edge)
            num_points_visual_regions_edges.append(len(visual_region_edge))

        # build padded array
        # make faster?
        # https://stackoverflow.com/questions/53071212/stacking-numpy-arrays-with-padding
        # https://stackoverflow.com/questions/53051560/stacking-numpy-arrays-of-different-length-using-padding/53052599?noredirect=1#comment93005810_53052599
        visual_regions_edges_array = np.zeros((num_interacting_nodes, np.max(num_points_visual_regions_edges), 2))
        for i in range(num_interacting_nodes):
            visual_region = visual_regions_edges[i]
            visual_regions_edges_array[i, :len(visual_region), :] = visual_region

    # ground-truth relations
    gt_relations = []
    # assign article_id to text_region based on most occuring text_lines
    num_tr_uncertain = 0
    tr_gt_article_ids = []
    for text_region in text_regions:
        # get all article_ids for textlines in this region
        tr_article_ids = []
        for text_line in text_region.text_lines:
            tr_article_ids.append(text_line.get_article_id())
        # count article_id occurences
        unique_article_ids = list(set(tr_article_ids))
        article_id_occurences = np.array([tr_article_ids.count(a_id) for a_id in unique_article_ids], dtype=np.int32)
        # assign article_id by majority vote
        if article_id_occurences.shape[0] > 1:
            num_tr_uncertain += 1
            assign_index = np.argmax(article_id_occurences)
            assign_id = unique_article_ids[int(assign_index)]
            tr_gt_article_ids.append(assign_id)
            logging.debug(f"TextRegion {text_region.id}: assign article_id '{assign_id}' (from {unique_article_ids})")
        else:
            tr_gt_article_ids.append(unique_article_ids[0])
    logging.debug(f"{num_tr_uncertain}/{len(text_regions)} text regions contained textlines of differing article_ids")
    # build gt ("1" means 'belong_to_same_article')
    for i, i_id in enumerate(tr_gt_article_ids):
        for j, j_id in enumerate(tr_gt_article_ids):
            if i_id == j_id:
                gt_relations.append([1, i, j])

    # number of ground-truth relations
    gt_num_relations = len(gt_relations)

    return np.array(num_nodes, dtype=np.int32), \
           interacting_nodes.astype(np.int32), \
           np.array(num_interacting_nodes, dtype=np.int32), \
           np.array(node_features, dtype=np.float32), \
           np.array(edge_features, dtype=np.float32) if edge_features else None, \
           np.transpose(np.array(visual_regions_nodes, dtype=np.float32), axes=(0, 2, 1)) if visual_regions else None, \
           np.array(num_points_visual_regions_nodes, dtype=np.int32) if visual_regions else None, \
           np.transpose(visual_regions_edges_array, axes=(0, 2, 1)) if visual_regions else None, \
           np.array(num_points_visual_regions_edges, dtype=np.int32) if visual_regions else None, \
           np.array(gt_relations, dtype=np.int32), \
           np.array(gt_num_relations, dtype=np.int32)


def generate_feature_jsons(page_paths,
                           out_path=None,
                           interaction="delaunay",
                           visual_regions=True,
                           json_list=None,
                           tb_similarity_setup=(None, None),
                           separators='line'):
    """
    Generates the input json files for a Graph Neural Network regarding the article separation task.

    For each pageXML file given in `page_paths` a corresponding json file will be generated, which contains the
    graph structure, node and edge features as well as the target relations.

    :param page_paths: list of pageXML file paths
    :param out_path: (optional) folder path to save the output to (defaults to a new 'json' folder besides the
        'page' folder where the pageXMl files are from)
    :param interaction: method for edge set generation ('delaunay' or 'fully')
    :param visual_regions: (bool) optionally build visual regions for nodes and edges (default False)
    :param json_list: (optional) list of additonal feature dictionaries from external json sources
    :param tb_similarity_setup: (optional) tuple ('language', 'wv_path'), where `language` is a string describing the
        underlying language of the word vector model given by `wv_path`
    :return: None
    """
    # Get external json data
    json_data = []
    if json_list:
        json_timer = time.time()
        for json_path in json_list:
            with open(json_path, "r") as json_file:
                json_data.append(json.load(json_file))
        logging.info(f"Time (loading external jsons): {time.time() - json_timer:.2f} seconds")

    # Setup textblock similarity feature extractor
    sim_feat_extractor = None
    if tb_similarity_setup[0] and tb_similarity_setup[1]:
        sim_feat_extractor = TextblockSimilarity(language=tb_similarity_setup[0], wv_path=tb_similarity_setup[1])

    # Get data from pagexml and write to json
    create_default_dir = False if out_path else True
    skipped_pages = []
    start_timer = time.time()
    for page_path in page_paths:
        logging.info(f"Processing... {page_path}")
        # build input & target
        num_nodes, interacting_nodes, num_interacting_nodes, node_features, edge_features, \
        visual_regions_nodes, num_points_visual_regions_nodes, \
        visual_regions_edges, num_points_visual_regions_edges, \
        gt_relations, gt_num_relations = \
            build_input_and_target(page_path=page_path,
                                   interaction=interaction,
                                   visual_regions=visual_regions,
                                   external_data=json_data,
                                   sim_feat_extractor=sim_feat_extractor,
                                   separators=separators)

        # build and write output
        if num_nodes is not None:
            out_dict = dict()
            out_dict["num_nodes"] = num_nodes.tolist()
            out_dict['interacting_nodes'] = interacting_nodes.tolist()
            out_dict['num_interacting_nodes'] = num_interacting_nodes.tolist()
            out_dict['node_features'] = node_features.tolist()
            out_dict['edge_features'] = edge_features.tolist()
            if visual_regions_nodes is not None and num_points_visual_regions_nodes is not None:
                out_dict['visual_regions_nodes'] = visual_regions_nodes.tolist()
                out_dict['num_points_visual_regions_nodes'] = num_points_visual_regions_nodes.tolist()
            if visual_regions_edges is not None and num_points_visual_regions_edges is not None:
                out_dict['visual_regions_edges'] = visual_regions_edges.tolist()
                out_dict['num_points_visual_regions_edges'] = num_points_visual_regions_edges.tolist()
            out_dict['gt_relations'] = gt_relations.tolist()
            out_dict['gt_num_relations'] = gt_num_relations.tolist()

            # Default output is a json folder one level above the pagexml file, indicating features and interaction
            if create_default_dir:
                visual = 'v' if visual_regions else ''
                out_path = re.sub(r'page$',
                                  f'json{node_features.shape[1]}{interaction[0]}{edge_features.shape[1]}{visual}{separators}',
                                  os.path.dirname(page_path))
            # Create output directory
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
                logging.info(f"Created output directory {out_path}")

            # Dump json
            file_name = os.path.splitext(os.path.basename(page_path))[0] + ".json"
            out = os.path.join(out_path, file_name)
            with open(out, "w") as out_file:
                json.dump(out_dict, out_file)
                logging.info(f"Saved json with graph features '{out}'")
        else:
            skipped_pages.append(page_path)
    logging.info(f"Time (feature generation): {time.time() - start_timer:.2f} seconds")
    logging.info(f"Wrote {len(page_paths) - len(skipped_pages)}/{len(page_paths)} files.")
    logging.info(f"Skipped {len(skipped_pages)} files:")
    for skipped in skipped_pages:
        logging.info(f"'{skipped}'")
