import copy

import jpype
from citlab_python_util.geometry.polygon import norm_poly_dists
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.geometry.util import check_intersection
from citlab_python_util.parser.xml.page.page_objects import TextLine

import matplotlib.patches as patches
from matplotlib import pyplot as plt
import numpy as np


class ArticleRectangle(Rectangle):

    def __init__(self, x=0, y=0, width=0, height=0, textlines=None, article_ids=None):
        """

        :type textlines: list of TextLine
        """
        super().__init__(x, y, width, height)

        self.textlines = textlines
        if article_ids is None and textlines is not None:
            self.a_ids = self.get_articles()
        else:
            self.a_ids = article_ids

    def get_articles(self):
        # traverse the baselines/textlines and check the article id
        article_set = set()

        for tl in self.textlines:
            article_set.add(tl.get_article_id())

        return article_set

    def contains_polygon(self, polygon, x, y, width, height):
        """ Checks if a polygon intersects with (or lies within) a (sub)rectangle given by the coordinates x,y
        (upper left point) and the width and height of the rectangle. """

        # iterate over the points of the polygon
        for i in range(polygon.n_points - 1):
            line_segment_bl = [polygon.x_points[i:i + 2], polygon.y_points[i:i + 2]]

            # The baseline segment lies outside the rectangle completely to the right/left/top/bottom
            if max(line_segment_bl[0]) <= x or min(line_segment_bl[0]) >= x + width or max(
                    line_segment_bl[1]) <= y or min(line_segment_bl[1]) >= y + height:
                continue

            # The baseline segment lies inside the rectangle
            if min(line_segment_bl[0]) >= x and max(line_segment_bl[0]) <= x + width and min(
                    line_segment_bl[1]) >= y and max(line_segment_bl[1]) <= y + height:
                return True

            # The baseline segment intersects with the rectangle or lies outside the rectangle but doesn't lie
            # completely right/left/top/bottom
            # First check intersection with the vertical line segments of the rectangle
            line_segment_rect_left = [[x, x], [y, y + height]]
            if check_intersection(line_segment_bl, line_segment_rect_left) is not None:
                return True
            line_segment_rect_right = [[x + width, x + width], [y, y + height]]
            if check_intersection(line_segment_bl, line_segment_rect_right) is not None:
                return True

            # Check other two sides
            line_segment_rect_top = [[x, x + width], [y, y]]
            if check_intersection(line_segment_bl, line_segment_rect_top) is not None:
                return True
            line_segment_rect_bottom = [[x, x + width], [y + height, y + height]]
            if check_intersection(line_segment_bl, line_segment_rect_bottom) is not None:
                return True

        return False

    def create_subregions_from_surrounding_polygon(self, ar_list=None, des_dist=5, max_d=50, max_rect_size=0):

        # width1 equals width2 if width is even, else width2 = width1 + 1
        # same for height1 and height2
        if ar_list is None:
            ar_list = []
        width1 = self.width // 2
        width2 = self.width - width1
        height1 = self.height // 2
        height2 = self.height - height1

        #########################
        #           #           #
        #     I     #     II    #
        #           #           #
        #########################
        #           #           #
        #    III    #     IV    #
        #           #           #
        #########################

        # determine textlines for each subregion
        tl1 = []
        tl2 = []
        tl3 = []
        tl4 = []
        bounds1 = []
        bounds2 = []
        bounds3 = []
        bounds4 = []
        a_ids1 = set()
        a_ids2 = set()
        a_ids3 = set()
        a_ids4 = set()

        # Get the non-overlapping bounding boxes for the algorithm
        tl_list = self.initialize_gt_generation(des_dist, max_d)

        for tl, tl_bound, tl_id in tl_list:

            intersection_rect = tl_bound.intersection(Rectangle(self.x, self.y, width1, height1))
            if intersection_rect.width > 0 and intersection_rect.height > 0:
                tl1 += [tl]
                bounds1 += [tl_bound]
                a_ids1.add(tl_id)

            intersection_rect = tl_bound.intersection(Rectangle(self.x + width1, self.y, width2, height1))
            if intersection_rect.width > 0 and intersection_rect.height > 0:
                tl2 += [tl]
                bounds2 += [tl_bound]
                a_ids2.add(tl_id)

            intersection_rect = tl_bound.intersection(Rectangle(self.x, self.y + height1, width1, height2))
            if intersection_rect.width > 0 and intersection_rect.height > 0:
                tl3 += [tl]
                bounds3 += [tl_bound]
                a_ids3.add(tl_id)

            intersection_rect = tl_bound.intersection(Rectangle(self.x + width1, self.y + height1, width2, height2))
            if intersection_rect.width > 0 and intersection_rect.height > 0:
                tl4 += [tl]
                bounds4 += [tl_bound]
                a_ids4.add(tl_id)

        a_rect1 = ArticleRectangle(self.x, self.y, width1, height1, tl1, a_ids1)
        a_rect2 = ArticleRectangle(self.x + width1, self.y, width2, height1, tl2, a_ids2)
        a_rect3 = ArticleRectangle(self.x, self.y + height1, width1, height2, tl3, a_ids3)
        a_rect4 = ArticleRectangle(self.x + width1, self.y + height1, width2, height2, tl4, a_ids4)

        # run create_subregions_from_surrounding_polygon on Rectangles that contain more than one TextLine object
        for a_rect in [a_rect1, a_rect2, a_rect3, a_rect4]:
            if len(a_rect.a_ids) > 1:
                a_rect.create_subregions_from_surrounding_polygon(ar_list, max_rect_size=max_rect_size)
            # TODO: height or width?
            elif 0 < max_rect_size < a_rect.height:
                a_rect.create_subregions_from_surrounding_polygon(ar_list, max_rect_size=max_rect_size)
            else:
                ar_list.append(a_rect)

        return ar_list

    def initialize_gt_generation(self, des_dist=5, max_d=50):
        # Create list of tuples containing the surrounding polygon, the baseline and the article id of each textline
        tl_list = []
        for tl in self.textlines:
            try:
                tl_bl = tl.baseline.to_polygon()
                tl_bl.calculate_bounds()
            except AttributeError:
                print(f"Textline with id {tl.id} has no baseline coordinates. Skipping...")
                continue

            tl_surr_poly = None
            try:
                tl_surr_poly = tl.surr_p.to_polygon().get_bounding_box()
            except (AttributeError, TypeError):
                print(f"Textline with id {tl.id} has no surrounding polygon.")

            tl_list.append([tl, tl_surr_poly, tl_bl, tl.get_article_id()])

        # Calculate the interline distance for each baseline
        # calculation of the normed polygons (includes also the calculation of their bounding boxes)
        list_of_normed_polygons = norm_poly_dists([tl[2] for tl in tl_list], des_dist=des_dist)

        # call java code to calculate the interline distances
        java_util = jpype.JPackage("citlab_article_separation.java").Util()

        list_of_normed_polygon_java = []

        for poly in list_of_normed_polygons:
            list_of_normed_polygon_java.append(jpype.java.awt.Polygon(poly.x_points, poly.y_points, poly.n_points))

        list_of_interline_distances_java = java_util.calcInterlineDistances(list_of_normed_polygon_java, des_dist,
                                                                            max_d)
        list_of_interline_distances = list(list_of_interline_distances_java)

        tl_list_copy = copy.deepcopy(tl_list)

        # Update the bounding boxes for the textlines
        for tl_tuple, tl_interdist in zip(tl_list_copy, list_of_interline_distances):
            _, _, tl_bl, _ = tl_tuple

            # bounding rectangle moved up and down
            height_shift = int(tl_interdist)
            tl_bl.bounds.translate(dx=0, dy=-height_shift)
            tl_bl.bounds.height += int(1.1 * height_shift)

        tl_surr_poly_final = []
        has_intersect_surr_polys = [False] * len(tl_list_copy)
        for i in range(len(tl_list_copy)):
            tl1, tl1_surr_poly, tl1_bl, tl1_aid = tl_list_copy[i]

            for j in range(i + 1, len(tl_list_copy)):
                tl2, tl2_surr_poly, tl2_bl, tl2_aid = tl_list_copy[j]

                def baseline_intersection_loop(bl1, bl2):
                    intersect = bl1.bounds.intersection(bl2.bounds)
                    while intersect.width >= 0 and intersect.height >= 0:

                        # TODO: Check if this works (bounding boxes intersect in a horizontal way)
                        if intersect.height == bl1.bounds.height or intersect.height == bl2.bounds.height:
                            width_shift = 1
                            # bl1 lies right of bl2
                            if bl1.bounds.x + bl1.bounds.width > bl2.bounds.x + bl2.bounds.width:
                                bl1.bounds.width -= width_shift
                                bl1.bounds.x += width_shift
                                bl2.bounds.width -= width_shift
                            # bl1 lies left of bl2
                            else:
                                bl1.bounds.width -= width_shift
                                bl2.bounds.x += width_shift
                                bl2.bounds.width -= width_shift

                        elif bl1.bounds.y + bl1.bounds.height > bl2.bounds.y + bl2.bounds.height:
                            height_shift = max(1, int(0.05 * bl1.bounds.height))

                            bl1.bounds.height -= height_shift
                            bl1.bounds.y += height_shift

                        elif bl2.bounds.y + bl2.bounds.height > bl1.bounds.y + bl1.bounds.height:
                            height_shift = max(1, int(0.05 * bl2.bounds.height))

                            bl2.bounds.height -= height_shift
                            bl2.bounds.y += height_shift

                        intersect = bl1.bounds.intersection(bl2.bounds)

                    return bl1

                if tl1_surr_poly is not None and not has_intersect_surr_polys[i]:
                    if tl2_surr_poly is not None and not has_intersect_surr_polys[j]:
                        intersection = tl1_surr_poly.intersection(tl2_surr_poly)
                        has_intersect_surr_polys[
                            j] = True if intersection.width >= 0 and intersection.height >= 0 else False
                    else:
                        intersection = tl1_surr_poly.intersection(tl2_bl.bounds)
                    if not (intersection.width >= 0 and intersection.height >= 0 and tl1_aid != tl2_aid):
                        if j == len(tl_list_copy) - 1:
                            tl_surr_poly_final.append((tl1, tl1_surr_poly, tl1_aid))
                        continue
                    has_intersect_surr_polys[i] = True
                else:
                    if tl2_surr_poly is not None:
                        intersection = tl1_bl.bounds.intersection(tl2_surr_poly)
                        has_intersect_surr_polys[
                            j] = True if intersection.width >= 0 and intersection.height >= 0 else False
                    else:
                        intersection = tl1_bl.bounds.intersection(tl2_bl.bounds)

                if intersection.width >= 0 and intersection.height >= 0 and tl1_aid != tl2_aid:
                    bl = baseline_intersection_loop(tl1_bl, tl2_bl)
                    if j == len(tl_list_copy) - 1:
                        tl_surr_poly_final.append((tl1, bl.bounds, tl1_aid))
                elif j == len(tl_list_copy) - 1:
                    tl_surr_poly_final.append((tl1, tl1_bl.bounds, tl1_aid))

        if len(has_intersect_surr_polys) > 0:
            if has_intersect_surr_polys[-1] or tl_list_copy[-1][1] is None:
                tl_surr_poly_final.append((tl_list_copy[-1][0], tl_list_copy[-1][2].bounds, tl_list_copy[-1][3]))
            else:
                tl_surr_poly_final.append((tl_list_copy[-1][0], tl_list_copy[-1][1], tl_list_copy[-1][3]))

        return tl_surr_poly_final


if __name__ == '__main__':
    # jpype.startJVM(jpype.getDefaultJVMPath())
    # tl1 = TextLine("tl_1", custom={"readingOrder": {"index": 0}, "structure": {"id": "a1", "type": "article"}},
    #                baseline=[(1, 3), (5, 1)])
    # tl2 = TextLine("tl_2", custom={"readingOrder": {"index": 1}, "structure": {"id": "a2", "type": "article"}},
    #                baseline=[(1, 4), (5, 2)])
    # tl3 = TextLine("tl_3", custom={"readingOrder": {"index": 2}, "structure": {"id": "a3", "type": "article"}},
    #                baseline=[(1, 5), (5, 5)])
    #
    # ar = ArticleRectangle(0, 0, 100, 100, [tl1, tl2, tl3])
    #
    # ar.create_subregions_from_surrounding_polygon()
    # jpype.shutdownJVM()

    fig, ax = plt.subplots()

    img = np.zeros([100, 100, 3], dtype=np.uint8)
    img.fill(0)

    ax.imshow(img)
    rect1 = patches.Rectangle((50, 50), 40, 30, linewidth=2, edgecolor='white', facecolor='none')
    rect2 = patches.Rectangle((40, 100), 40, 30, linewidth=2, edgecolor='white', facecolor='none')
    ax.add_patch(rect1)
    ax.add_patch(rect2)
    # plt.plot_binary(plt.Rectangle((1, 1), 50, 50))
    plt.show()
