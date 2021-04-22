import copy
from collections import defaultdict

from citlab_python_util.geometry.polygon import Polygon, list_to_polygon_object
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.geometry.util import ortho_connect, smooth_surrounding_polygon, polygon_clip, convex_hull, \
    bounding_box, merge_rectangles
from citlab_python_util.image_processing.white_space_detection import get_binarization, is_whitespace
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.page_objects import Points

from citlab_article_separation.article_rectangle import ArticleRectangle


def get_article_surrounding_polygons(ar_dict):
    """
    Create surrounding polygons over sets of rectangles, belonging to different article_ids.

    :param ar_dict: dict (keys = article_id, values = corresponding rectangles)
    :return: dict (keys = article_id, values = corresponding surrounding polygons)
    """
    asp_dict = {}
    for id in ar_dict:
        sp = ortho_connect(ar_dict[id])
        asp_dict[id] = sp
    return asp_dict


def smooth_article_surrounding_polygons(asp_dict, poly_norm_dist=10, orientation_dims=(600, 300, 600, 300), offset=0):
    """
    Create smoothed polygons over "crooked" polygons, belonging to different article_ids.

    1.) The polygon gets normalized, where the resulting vertices are at most `poly_norm_dist` pixels apart.

    2.) For each vertex of the original polygon an orientation is determined:

    2.1) Four rectangles (North, East, South, West) are generated, with the dimensions given by `or_dims`
    (width_vertical, height_vertical, width_horizontal, height_horizontal), i.e. North and South rectangles
    have dimensions width_v x height_v, whereas East and West rectangles have dimensions width_h x height_h.

    2.2) The offset controls how far the cones overlap (e.g. how far the north cone gets translated south)

    2.3) Each rectangle counts the number of contained points from the normalized polygon

    2.4) The top two rectangle counts determine the orientation of the vertex: vertical, horizontal or one
    of the four possible corner types.

    3.) Vertices with a differing orientation to its agreeing neighbours are assumed to be mislabeled and
    get its orientation converted to its neighbours.

    4.) Corner clusters of the same type need to be shrunken down to one corner, with the rest being converted
    to verticals. (TODO or horizontals)

    5.) Clusters between corners (corner-V/H-...-V/H-corner) get smoothed if they contain at least five points,
    by taking the average over the y-coordinates for horizontal edges and the average over the x-coordinates for
    vertical edges.

    :param asp_dict: dict (keys = article_id, values = list of "crooked" polygons)
    :param poly_norm_dist: int, distance between pixels in normalized polygon
    :param orientation_dims: tuple (width_v, height_v, width_h, height_h), the dimensions of the orientation rectangles
    :param offset: int, number of pixel that the orientation cones overlap
    :return: dict (keys = article_id, values = smoothed polygons)
    """
    asp_dict_smoothed = {}
    for id in asp_dict:
        asp_dict_smoothed[id] = []
        for poly in asp_dict[id]:
            sp_smooth = smooth_surrounding_polygon(poly, poly_norm_dist, orientation_dims, offset)
            asp_dict_smoothed[id].append(sp_smooth)
    return asp_dict_smoothed


def convert_blank_article_rects_by_rects(ars_dict, method="bb"):
    assert method == "bb" or method == "ch", "Only supports methods 'bb' (bounding boxes) and 'ch' (convex hulls)"
    # Build up bounding boxes / convex hulls over rectangle vertices
    poly_dict = {}
    for key in ars_dict:
        if key == "blank" or key is None:
            continue
        article_point_set = []
        for ar in ars_dict[key]:
            article_point_set += ar.get_vertices()
        if method == "bb":
            poly_dict[key] = bounding_box(article_point_set)
        elif method == "ch":
            poly_dict[key] = convex_hull(article_point_set)

    out_dict = ars_dict.copy()
    to_remove = []
    # Go over blank rectangles and check for intersections with other articles
    for ar in ars_dict["blank"]:
        intersections = []
        for key in poly_dict:
            if polygon_clip(ar.get_vertices(), poly_dict[key]):
                intersections.append(key)
        # got exactly 1 intersection
        if len(intersections) == 1:
            # Convert rectangle to respective article id
            out_dict[intersections[0]].append(ar)
            to_remove.append(ar)
    # Remove relevant rectangles from blanks
    out_dict["blank"] = [ar for ar in ars_dict["blank"] if ar not in to_remove]
    return out_dict


def convert_blank_article_rects_by_polys(ars_dict, asp_dict, method="bb"):
    assert method == "bb" or method == "ch", "Only supports methods 'bb' (bounding boxes) and 'ch' (convex hulls)"
    # Build up bounding boxes / convex hulls over polygon vertices
    poly_dict = {}
    for key in asp_dict:
        if key == "blank" or key is None:
            continue
        poly_dict[key] = []
        for sp in asp_dict[key]:
            if method == "bb":
                poly_dict[key].append(bounding_box(sp.as_list()))
            elif method == "ch":
                poly_dict[key].append(convex_hull(sp.as_list()))

    out_dict = ars_dict.copy()
    to_remove = []
    # Go over blank rectangles and check for intersections with other articles
    for ar in ars_dict["blank"]:
        intersections = []
        for key in poly_dict:
            for poly in poly_dict[key]:
                if polygon_clip(ar.get_vertices(), poly):
                    intersections.append(key)
        # got exactly 1 intersection
        print("AR: {}".format(ar.get_vertices()))
        print("Intersections: {}".format(intersections))
        if len(set(intersections)) == 1:
            # Convert rectangle to respective article id
            out_dict[intersections[0]].append(ar)
            to_remove.append(ar)
    # Remove relevant rectangles from blanks
    out_dict["blank"] = [ar for ar in ars_dict["blank"] if ar not in to_remove]
    return out_dict


def is_vertical_aligned(line1, line2, margin=20):
    line1_min_x = min(line1, key=lambda point: point[0])[0]
    line1_max_x = max(line1, key=lambda point: point[0])[0]
    line2_min_x = min(line2, key=lambda point: point[0])[0]
    line2_max_x = max(line2, key=lambda point: point[0])[0]

    if line2_min_x - margin <= line1_min_x <= line2_max_x and line2_min_x <= line1_max_x <= line2_max_x + margin:
        return True

    if line1_min_x - margin <= line2_min_x <= line1_max_x and line1_min_x <= line2_max_x <= line1_max_x + margin:
        return True

    if line1_min_x - margin < line2_min_x < line1_min_x + margin or line1_max_x - margin < line2_max_x < line1_max_x + margin:
        return True

    return False


def sort_textlines_by_y(textlines):
    return sorted(textlines, key=lambda textline: min(textline.baseline.points_list, key=lambda point: point[1])[1])


def stretch_rectangle_until_whitespace(binarized_image, rectangle, whitespace_height=1, stretch_limit=250):
    """

    :type rectangle: Rectangle
    """
    new_rectangle = copy.deepcopy(rectangle)
    # whitespace_rectangle = Rectangle(x=rectangle.x, y=rectangle.y - whitespace_height, width=rectangle.width,
    #                                  height=whitespace_height)
    whitespace_rectangle = Rectangle(x=rectangle.x + rectangle.width // 5, y=rectangle.y - whitespace_height,
                                     width=3 * rectangle.width // 5,
                                     height=whitespace_height)

    if whitespace_rectangle.y < 0 or whitespace_rectangle.y + whitespace_rectangle.height > binarized_image.shape[1]:
        return new_rectangle

    for i in range(stretch_limit):
        if is_whitespace(binarized_image, whitespace_rectangle, threshold=0.04) or whitespace_rectangle.y == 0:
            new_rectangle.set_bounds(rectangle.x, whitespace_rectangle.y, rectangle.width,
                                     rectangle.height + i + 1)
            break
        else:
            whitespace_rectangle.translate(0, -1)

    return new_rectangle


# TODO: Optimize code
def get_article_rectangles_from_baselines(page, image_path, stretch=False, use_surr_polygons=True):
    if type(page) == str:
        page = Page(page)

    assert type(page) == Page, f"Type must be Page, got {type(page)} instead."
    article_dict = page.get_article_dict()

    article_rectangles_dict = defaultdict(list)

    if stretch:
        binarized_image = get_binarization(image_path)

    for article_id, textlines in article_dict.items():
        used_textline_ids = []
        sorted_textlines = sort_textlines_by_y(textlines)
        for i, textline in enumerate(sorted_textlines):
            # used_textline_ids = [tl.id for article_rectangle in article_rectangles_dict[article_id] for tl in
            #                      article_rectangle.textlines]
            if textline.id in used_textline_ids:
                continue

            baseline = textline.baseline.points_list
            baseline_polygon = textline.baseline.to_polygon()

            if use_surr_polygons:
                baseline_bounding_box = textline.surr_p.to_polygon().get_bounding_box() if textline.surr_p else baseline_polygon.get_bounding_box()
            else:
                baseline_bounding_box = baseline_polygon.get_bounding_box()

            # [ar for aid, ar in article_rectangles_dict.items() if aid != article_id]

            # print(baseline_bounding_box.get_vertices())
            # print(article_id)
            for ars in [ar for aid, ar in article_rectangles_dict.items() if aid != article_id]:
                for ar in ars:
                    intersection = ar.intersection(baseline_bounding_box)
                    for _ in range(20):
                        if intersection.width > 0 and intersection.height > 0:
                            baseline_bounding_box.translate(0, 1)
                            baseline_bounding_box.height -= 1
                            intersection = ar.intersection(baseline_bounding_box)
                        else:
                            break

            article_rectangle = ArticleRectangle(baseline_bounding_box.x, baseline_bounding_box.y,
                                                 baseline_bounding_box.width, baseline_bounding_box.height,
                                                 [textline], None)

            used_textline_ids.append(textline.id)
            if i == len(sorted_textlines):
                continue
            for j, textline_compare in enumerate(sorted_textlines[i + 1:]):
                if textline_compare.id in used_textline_ids:
                    continue
                # for tl in article_rectangle.textlines:
                #     print(tl.baseline.points_list)
                baseline_compare = textline_compare.baseline.points_list
                skip = False

                # instead of checking whether the two baselines are aligned, we should check, if the current article
                # rectangle and the baseline_compare are aligned!
                article_rectangle_horizontal_poly = article_rectangle.get_vertices()[:2]

                # if not is_vertical_aligned(baseline, baseline_compare):
                if not is_vertical_aligned(article_rectangle_horizontal_poly, baseline_compare):
                    if i + j + 2 != len(sorted_textlines):
                        for tl in sorted_textlines[i + j + 2:]:
                            if tl.id not in used_textline_ids:
                                if is_vertical_aligned(baseline, tl.baseline.points_list) and is_vertical_aligned(
                                        baseline_compare, tl.baseline.points_list, margin=50):
                                    skip = False
                                    break
                                else:
                                    skip = True
                    else:
                        skip = True
                if skip:
                    continue

                baseline_compare_polygon = textline_compare.baseline.to_polygon()
                if use_surr_polygons:
                    baseline_compare_bounding_box = textline_compare.surr_p.to_polygon().get_bounding_box() if textline_compare.surr_p else baseline_compare_polygon.get_bounding_box()
                else:
                    baseline_compare_bounding_box = baseline_compare_polygon.get_bounding_box()

                merged_rectangle = merge_rectangles([article_rectangle, baseline_compare_bounding_box])

                skip = False
                for ars in article_rectangles_dict.values():
                    for ar in ars:
                        intersection = ar.intersection(merged_rectangle)
                        if intersection.width > 0 and intersection.height > 0:
                            skip = True
                            break
                    if skip:
                        break
                if skip:
                    continue

                merged_article_rectangle = ArticleRectangle(merged_rectangle.x, merged_rectangle.y,
                                                            merged_rectangle.width, merged_rectangle.height)
                # if merged_article_rectangle contains any other baseline, that is not yet in an article_rectangle, skip
                # textlines_to_check_intersection = [tl for tl in sorted_textlines if
                #                                    tl.id not in used_textline_ids and tl.id != textline_compare.id]
                textlines_to_check_intersection = []
                textlines_to_check_intersection += [tl for textlines in
                                                    [article_dict[aid] for aid in article_dict if
                                                     aid != article_id] for tl in textlines]
                # polygons_to_check_intersection = [tl.surr_p.to_polygon() if tl.surr_p is not None else
                #                                   tl.baseline.to_polygon() for tl in textlines_to_check_intersection]
                polygons_to_check_intersection = [tl.baseline.to_polygon() for tl in textlines_to_check_intersection]

                skip = False
                for polygon in polygons_to_check_intersection:
                    if merged_article_rectangle.contains_polygon(polygon, merged_article_rectangle.x,
                                                                 merged_article_rectangle.y,
                                                                 merged_article_rectangle.width,
                                                                 merged_article_rectangle.height):
                        skip = True

                        merged_article_rectangle_copy = copy.deepcopy(merged_article_rectangle)
                        for _ in range(50):
                            merged_article_rectangle_copy.translate(0, 1)
                            merged_article_rectangle_copy.height -= 1
                            if not merged_article_rectangle_copy.contains_polygon(polygon,
                                                                                  merged_article_rectangle_copy.x,
                                                                                  merged_article_rectangle_copy.y,
                                                                                  merged_article_rectangle_copy.width,
                                                                                  merged_article_rectangle_copy.height):
                                skip = False
                            merged_article_rectangle = merged_article_rectangle_copy
                            break

                    if skip:
                        break

                if skip:
                    continue

                article_rectangle.textlines.append(textline_compare)
                article_rectangle.set_bounds(merged_article_rectangle.x, merged_article_rectangle.y,
                                             merged_article_rectangle.width, merged_article_rectangle.height)
                used_textline_ids.append(textline_compare.id)

            if len(article_rectangle.textlines) == 1:
                if article_rectangle.textlines[0].surr_p:
                    # bb = article_rectangle.textlines[0].surr_p.to_polygon().get_bounding_box()
                    # article_rectangle.set_bounds(bb.x, bb.y, bb.width, bb.height)
                    pass
                else:
                    article_rectangle.translate(0, -10)
                    article_rectangle.height = 10

            if stretch:
                img_height = len(binarized_image)
                article_rectangle = stretch_rectangle_until_whitespace(binarized_image, article_rectangle,
                                                                       whitespace_height=max(1, img_height // 1000),
                                                                       stretch_limit=img_height // 10)

            article_rectangles_dict[article_id].append(article_rectangle)

    return article_rectangles_dict


def merge_article_rectangles_vertically(article_rectangles_dict, min_width_intersect=20, max_vertical_distance=50, use_convex_hull=False):
    """

    :type article_rectangles_dict: dict[str,list[ArticleRectangle]]
    """
    surr_polygon_dict = defaultdict(list)

    for aid, article_rectangles_list in article_rectangles_dict.items():
        redundant_article_rectangles = []
        merged_articles_list = []
        for i, article_rectangle in enumerate(article_rectangles_list):
            if article_rectangle in redundant_article_rectangles:
                continue
            merged_articles = [article_rectangle]
            for l in merged_articles_list:
                if article_rectangle in l:
                    merged_articles_list.remove(l)
                    merged_articles = l
                    break

            if i + 1 == len(article_rectangles_list):
                merged_articles_list.append(merged_articles)
                break
            for article_rectangle_compare in article_rectangles_list[i + 1:]:
                if article_rectangle_compare in redundant_article_rectangles:
                    continue
                skip = False
                if article_rectangle.contains_rectangle(article_rectangle_compare):
                    # no need to add article rectangle, since it gives no new information
                    redundant_article_rectangles.append(article_rectangle_compare)
                    continue
                intersection = article_rectangle.intersection(article_rectangle_compare)
                if intersection.width > min_width_intersect and intersection.height > 0:
                    # TODO: Check intersection with other rectangle of same aid?
                    merged_articles.append(article_rectangle_compare)
                    merged_articles.append(intersection)

                if intersection.width > min_width_intersect and intersection.height < 0:
                    if abs(intersection.height) < max_vertical_distance:
                        gap = article_rectangle.get_gap_to(article_rectangle_compare)
                        # check if there is an intersection with another article rectangle in this area
                        for ar in [_ar for _ars in article_rectangles_dict.values() for _ar in _ars if
                                   _ar != article_rectangle]:
                            intersection_gap_with_rectangle = gap.intersection(ar)
                            if intersection_gap_with_rectangle.height > 0 and intersection_gap_with_rectangle.width > 0:
                                skip = True
                                break
                        if skip:
                            continue
                        merged_articles.append(article_rectangle_compare)
                        merged_articles.append(gap)

            merged_articles_list.append(merged_articles)
        if use_convex_hull:
            for _ars in merged_articles_list:
                article_convex_hull = convex_hull(
                    [vertex for vertices in [_ar.get_vertices() for _ar in _ars] for vertex in vertices])
                article_convex_hull_polygon = list_to_polygon_object(article_convex_hull)
                surr_polygon_dict[aid].append(article_convex_hull_polygon)
        else:
            for _ars in merged_articles_list:
                article_ortho_connect_polygon = ortho_connect(_ars)
                for ortho_connect_polygon in article_ortho_connect_polygon:
                    surr_polygon_dict[aid].append(ortho_connect_polygon)

    return surr_polygon_dict


def get_article_rectangles_from_surr_polygons(page, use_max_rect_size=True, max_d=0, max_rect_size_scale=1 / 50,
                                              max_d_scale=1 / 20):
    """Given the PageXml file `page` return the corresponding article subregions as a list of ArticleRectangle objects.
     Also returns the width and height of the image (NOT of the PrintSpace).

    :param page: Either the path to the PageXml file or a Page object.
    :type page: Union[str, Page]
    :param use_max_rect_size: whether to use a max rectangle size for the article rectangles or not
    :type use_max_rect_size: bool
    :return: the article subregion list, the height and the width of the image
    """
    if type(page) == str:
        page = Page(page)

    assert type(page) == Page, f"Type must be Page, got {type(page)} instead."
    ps_coords = page.get_print_space_coords()
    ps_poly = Points(ps_coords).to_polygon()
    # Maybe check if the surrounding Rectangle of the polygon has corners given by ps_poly
    ps_rectangle = ps_poly.get_bounding_box()

    # First ArticleRectangle to consider
    ps_rectangle = ArticleRectangle(ps_rectangle.x, ps_rectangle.y, ps_rectangle.width, ps_rectangle.height,
                                    page.get_textlines())

    if use_max_rect_size:
        max_rect_size = int(max_rect_size_scale * ps_rectangle.height)
    else:
        max_rect_size = 0
    if not max_d:
        max_d = int(max_d_scale * ps_rectangle.height)

    ars = ps_rectangle.create_subregions_from_surrounding_polygon(max_d=max_d, max_rect_size=max_rect_size)

    # ars = ps_rectangle.create_subregions_from_surrounding_polygon(max_d=int(1 / 20 * ps_rectangle.height))

    img_width, img_height = page.get_image_resolution()

    return ars, img_height, img_width


if __name__ == '__main__':
    # xml_path = "/home/max/data/as/NewsEye_ONB_data_corrected/aze/ONB_aze_18950706_corrected/page/ONB_aze_18950706_4.xml"
    # img_path = "/home/max/data/as/NewsEye_ONB_data_corrected/aze/ONB_aze_18950706_corrected/ONB_aze_18950706_4.jpg"
    # xml_path = "/home/max/data/as/NewsEye_ONB_data_corrected/krz/ONB_krz_19110701_corrected/page/ONB_krz_19110701_016.xml"
    # img_path = "/home/max/data/as/NewsEye_ONB_data_corrected/krz/ONB_krz_19110701_corrected/ONB_krz_19110701_016" \
    #            ".jpg"
    # #
    img_path = "/home/max/devel/projects/article_separation/data/newseye_onb/ibn/ONB_ibn_18640702_corrected/ONB_ibn_18640702_003.tif"
    xml_path = "/home/max/devel/projects/article_separation/data/newseye_onb/ibn/ONB_ibn_18640702_corrected/page/ONB_ibn_18640702_003.xml"

    # xml_path = "/home/max/data/as/NewsEye_ONB_data_corrected/ibn/ONB_ibn_19330701_corrected/page/ONB_ibn_19330701_001.xml"
    # img_path = "/home/max/data/as/NewsEye_ONB_data_corrected/ibn/ONB_ibn_19330701_corrected/ONB_ibn_19330701_001.jpg"
    # # #
    # xml_path = "/home/max/data/as/NewsEye_ONB_data_corrected/nfp/ONB_nfp_18730705_corrected/page/ONB_nfp_18730705_016.xml"
    # img_path = "/home/max/data/as/NewsEye_ONB_data_corrected/nfp/ONB_nfp_18730705_corrected/ONB_nfp_18730705_016.tif"
    #
    # xml_path = '/home/max/data/as/NewsEye_ONB_data_corrected/nfp/ONB_nfp_18950706_corrected/page/ONB_nfp_18950706_015.xml'
    # img_path = '/home/max/data/as/NewsEye_ONB_data_corrected/nfp/ONB_nfp_18950706_corrected/ONB_nfp_18950706_015.tif'

    article_rectangles_dict = get_article_rectangles_from_baselines(Page(xml_path), img_path, use_surr_polygons=True,
                                                                    stretch=False)

    surr_polys_dict = merge_article_rectangles_vertically(article_rectangles_dict)

    import matplotlib.pyplot as plt
    from citlab_python_util.parser.xml.page import plot as page_plot
    from matplotlib.collections import PolyCollection
    from citlab_python_util.plot import colors

    # page_plot.plot_pagexml(xml_path, img_path)

    fig, ax = plt.subplots()
    page_plot.add_image(ax, img_path)

    for i, a_id in enumerate(surr_polys_dict):
        surr_polygons = surr_polys_dict[a_id]
        if a_id is None:
            surr_poly_collection = PolyCollection([surr_poly.as_list() for surr_poly in surr_polygons], closed=True,
                                                  edgecolors=colors.DEFAULT_COLOR, facecolors=colors.DEFAULT_COLOR)
        else:
            surr_poly_collection = PolyCollection([surr_poly.as_list() for surr_poly in surr_polygons], closed=True,
                                                  edgecolors=colors.COLORS[i], facecolors=colors.COLORS[i])
        surr_poly_collection.set_alpha(0.5)
        ax.add_collection(surr_poly_collection)

    # plt.show()

    fig, ax = plt.subplots()
    page_plot.add_image(ax, img_path)

    for i, a_id in enumerate(article_rectangles_dict):
        # fig, ax = plt.subplots()
        # page_plot.add_image(ax, img_path)
        # add facecolors="None" if rectangles should not be filled
        ars = article_rectangles_dict[a_id]
        if a_id is None:
            ar_poly_collection = PolyCollection([ar.get_vertices() for ar in ars], closed=True,
                                                edgecolors=colors.DEFAULT_COLOR, facecolors=colors.DEFAULT_COLOR)
        else:
            ar_poly_collection = PolyCollection([ar.get_vertices() for ar in ars], closed=True,
                                                edgecolors=colors.COLORS[i], facecolors=colors.COLORS[i])
        ar_poly_collection.set_alpha(0.5)
        ax.add_collection(ar_poly_collection)

        for ar in ars:
            if ar.height == 0:
                print(ar.width, ar.height, len(ar.textlines))
                for textline in ar.textlines:
                    print("\t", textline.baseline.points_list)

        # plt.show()

    plt.show()

    # for aid, ars in article_rectangles_dict.items():
    #     print(aid)
    #     print(len(ars))
    #     for ar in ars:
    #         print('\t', ar.get_vertices())

    # print(article_rectangles_dict)
