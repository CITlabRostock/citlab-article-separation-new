import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import jpype

from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.basic.list_util import filter_by_attribute
from citlab_article_separation.util import get_article_rectangles_from_surr_polygons, smooth_article_surrounding_polygons, \
    convert_blank_article_rects_by_rects, \
    convert_blank_article_rects_by_polys
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.geometry.util import ortho_connect, convex_hull, bounding_box
from citlab_python_util.parser.xml.page import plot as page_plot
from citlab_python_util.plot import colors


def plot_gt_data(img_path, poly_dict, poly_smooth_dict, show=True):
    fig, axs = plt.subplots(ncols=2)
    page_plot.add_image(axs[0], img_path)
    page_plot.add_image(axs[1], img_path)
    if poly_dict:
        for i, j in enumerate(poly_dict):
            # add facecolors="None" if rectangles should not be filled
            polys = poly_dict[j]
            if j == "blank":
                ar_poly_collection = PolyCollection(polys, closed=True, edgecolors='k', facecolors='k')
            else:
                ar_poly_collection = PolyCollection(polys, closed=True, edgecolors=colors.COLORS[i],
                                                    facecolors=colors.COLORS[i])
            ar_poly_collection.set_alpha(0.5)
            axs[0].add_collection(ar_poly_collection)

    if poly_smooth_dict:
        for i, j in enumerate(poly_smooth_dict):
            # add facecolors="None" if rectangles should not be filled
            polys = poly_smooth_dict[j]
            if j == "blank":
                continue
                # ar_poly_collection = PolyCollection(polys, closed=True, edgecolors='k', facecolors='k')
            else:
                ar_poly_collection = PolyCollection(polys, closed=True, edgecolors=colors.COLORS[i],
                                                    facecolors=colors.COLORS[i])
            ar_poly_collection.set_alpha(0.5)
            axs[1].add_collection(ar_poly_collection)

    if show:
        plt.show()


if __name__ == '__main__':
    jpype.startJVM(jpype.getDefaultJVMPath())
    # path_to_page_xml = "/home/johannes/devel/projects/as/ArticleSeparationMeasure/test/resources/" \
    #                    "newseye_as_test_data/xml_files_gt/19000715_1-0003.xml"
    # path_to_img = "/home/johannes/devel/projects/as/ArticleSeparationMeasure/test/resources/" \
    #               "newseye_as_test_data/image_files/19000715_1-0003.jpg"

    path_to_page_xml = '/home/max/devel/projects/article_separation/data/newseye_onb/aze/ONB_aze_18950706_corrected/' \
                       'page/ONB_aze_18950706_5.xml'
    path_to_img = '/home/max/devel/projects/article_separation/data/newseye_onb/aze/ONB_aze_18950706_corrected/' \
                  'ONB_aze_18950706_5.jpg'

    # path_to_page_xml = "/home/johannes/devel/projects/as/ArticleSeparationMeasure/test/resources/" \
    #                    "newseye_as_test_data/xml_files_gt/19420115_1-0002.xml"
    # path_to_img = "/home/johannes/devel/projects/as/ArticleSeparationMeasure/test/resources/" \
    #               "newseye_as_test_data/image_files/19420115_1-0002.jpg"

    path_to_page_xml = path_to_page_xml.strip()
    page = Page(path_to_page_xml)

    # Get the article rectangles as a list of ArticleRectangle objects
    ars, img_height, img_width = get_article_rectangles_from_surr_polygons(page)

    # resize the image to draw the border polygons (if available)
    img_height += 1
    img_width += 1
    print("img_height = {}, img_width = {}".format(img_height, img_width))

    # Convert the list of article rectangles to a dictionary with the article ids as keys
    # and the corresponding list of rectangles as value
    ars_dict = filter_by_attribute(ars, "a_ids")
    print("Len(Blank) = ", len(ars_dict["blank"]))

    # Convert blank article rectangles (by rectangles)
    # ... over bounding boxes
    ars_dict = convert_blank_article_rects_by_rects(ars_dict, method="bb")
    print("Len(Blank) = ", len(ars_dict["blank"]))
    # ... over convex hulls
    ars_dict = convert_blank_article_rects_by_rects(ars_dict, method="ch")
    print("Len(Blank) = ", len(ars_dict["blank"]))

    # Convert the article rectangles to surrounding polygons
    surr_polys_dict = {}
    for a_id, ars_sub in ars_dict.items():
        # if a_id == 'blank':
        #     continue
        rs = [Rectangle(ar.x, ar.y, ar.width, ar.height) for ar in ars_sub]
        surr_polys = ortho_connect(rs)
        surr_polys_dict[a_id] = surr_polys

    # # # Print it
    # plot_dict = {}
    # for k in surr_polys_dict:
    #     plot_dict[k] = []
    #     for poly in surr_polys_dict[k]:
    #         plot_dict[k].append(poly.as_list())
    # plot_gt_data(path_to_img, plot_dict, None, show=False)

    # TODO: Check blank rectangles for correct intersections
    # TODO: More should get converted here imo?!
    # Convert blank article rectangles (by polygons)
    # ... over bounding boxes
    ars_dict = convert_blank_article_rects_by_polys(ars_dict, surr_polys_dict, method="bb")
    print("Len(Blank) = ", len(ars_dict["blank"]))
    # ... over convex hulls
    ars_dict = convert_blank_article_rects_by_polys(ars_dict, surr_polys_dict, method="ch")
    print("Len(Blank) = ", len(ars_dict["blank"]))

    # Convert the article rectangles to surrounding polygons (again)
    surr_polys_dict = {}
    for a_id, ars_sub in ars_dict.items():
        # if a_id == 'blank':
        #     continue
        rs = [Rectangle(ar.x, ar.y, ar.width, ar.height) for ar in ars_sub]
        surr_polys = ortho_connect(rs)
        surr_polys_dict[a_id] = surr_polys

    or_dims = [700, 400, 700, 400]
    poly_norm = 5
    offset = 0
    surr_polys_smooth_dict = smooth_article_surrounding_polygons(surr_polys_dict, poly_norm, or_dims, offset)

    # # Print it
    plot_dict = {}
    for k in surr_polys_dict:
        plot_dict[k] = []
        for poly in surr_polys_dict[k]:
            plot_dict[k].append(poly.as_list())

    plot_dict_smooth = {}
    for k in surr_polys_smooth_dict:
        plot_dict_smooth[k] = []
        for poly in surr_polys_smooth_dict[k]:
            plot_dict_smooth[k].append(poly.as_list())

    plot_gt_data(path_to_img, plot_dict, plot_dict_smooth, show=True)

    # for id in surr_polys_smooth_dict:
    #     plt.figure()
    #     plt.suptitle("Article '" + str(id) + "'")
    #     rect_points = []
    #     for rect in ars_dict[id]:
    #         vertices = rect.get_vertices()
    #         vertices.append(vertices[0])
    #         rect_points.append(vertices)
    #
    #     plt.subplot(1, 3, 1)
    #     for poly in rect_points:
    #         xs, ys = zip(*poly)
    #         plt.ylim(-10000, 1000)
    #         plt.xlim(-1000, 7000)
    #         plt.plot_binary(xs, [-y for y in ys], 'k')
    #
    #     plt.subplot(1, 3, 2)
    #     for poly in surr_polys_dict[id]:
    #         poly = poly.as_list()
    #         poly.append(poly[0])
    #         xs, ys = zip(*poly)
    #         plt.ylim(-10000, 1000)
    #         plt.xlim(-1000, 7000)
    #         plt.plot_binary(xs, [-y for y in ys], 'k')
    #
    #     smooth_corners_ur_x = []
    #     smooth_corners_ur_y = []
    #     smooth_corners_ul_x = []
    #     smooth_corners_ul_y = []
    #     smooth_corners_dr_x = []
    #     smooth_corners_dr_y = []
    #     smooth_corners_dl_x = []
    #     smooth_corners_dl_y = []
    #     smooth_verticals_x = []
    #     smooth_verticals_y = []
    #     smooth_horizontals_x = []
    #     smooth_horizontals_y = []
    #
    #     # for poly in surr_polys_smooth_dict[id]:
    #     #     rects = []
    #     #     print(poly)
    #     #     for idx, pt in enumerate(poly):
    #     #         if 'corner' in pt[1]:
    #     #             if 'ur' in pt[1]:
    #     #                 smooth_corners_ur_x.append(pt[0][0])
    #     #                 smooth_corners_ur_y.append(pt[0][1])
    #     #             elif 'ul' in pt[1]:
    #     #                 smooth_corners_ul_x.append(pt[0][0])
    #     #                 smooth_corners_ul_y.append(pt[0][1])
    #     #             elif 'dr' in pt[1]:
    #     #                 smooth_corners_dr_x.append(pt[0][0])
    #     #                 smooth_corners_dr_y.append(pt[0][1])
    #     #             else:
    #     #                 smooth_corners_dl_x.append(pt[0][0])
    #     #                 smooth_corners_dl_y.append(pt[0][1])
    #     #             print_rects = False
    #     #             if print_rects and 'corner' in poly[idx][1]:
    #     #                     orientation_rects = get_orientation_rectangles(pt[0], 400, 800, 600, 400).values()
    #     #                     for o_rect in orientation_rects:
    #     #                         o_vertices = o_rect.get_vertices()
    #     #                         o_vertices.append(o_vertices[0])
    #     #                         rects.append(o_vertices)
    #     #         elif pt[1] == 'vertical':
    #     #             smooth_verticals_x.append(pt[0][0])
    #     #             smooth_verticals_y.append(pt[0][1])
    #     #         elif pt[1] == 'horizontal':
    #     #             smooth_horizontals_x.append(pt[0][0])
    #     #             smooth_horizontals_y.append(pt[0][1])
    #     #     for poly in rects:
    #     #         xs, ys = zip(*poly)
    #     #         plt.plot_binary(xs, [-y for y in ys], 'r')
    #     # plt.plot_binary(smooth_corners_ur_x, [-y for y in smooth_corners_ur_y], 'xb')
    #     # plt.plot_binary(smooth_corners_ul_x, [-y for y in smooth_corners_ul_y], 'xg')
    #     # plt.plot_binary(smooth_corners_dr_x, [-y for y in smooth_corners_dr_y], 'xr')
    #     # plt.plot_binary(smooth_corners_dl_x, [-y for y in smooth_corners_dl_y], 'xc')
    #     # plt.plot_binary(smooth_verticals_x, [-y for y in smooth_verticals_y], 'om', ms=3)
    #     # plt.plot_binary(smooth_horizontals_x, [-y for y in smooth_horizontals_y], 'oy', ms=4)
    #
    #     plt.subplot(1, 3, 3)
    #     for poly in surr_polys_smooth_dict[id]:
    #         poly = poly.as_list()
    #         poly.append(poly[0])
    #         xs, ys = zip(*poly)
    #         plt.ylim(-10000, 1000)
    #         plt.xlim(-1000, 7000)
    #         plt.plot_binary(xs, [-y for y in ys], 'k')
    #
    # plt.show()
    jpype.shutdownJVM()
