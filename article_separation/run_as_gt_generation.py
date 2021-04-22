import os
from argparse import ArgumentParser, ArgumentTypeError

import cv2
import jpype
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from citlab_python_util.geometry.point import rescale_points
from citlab_python_util.image_processing.morphology import apply_transform
from citlab_python_util.parser.xml.page import plot as page_plot
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.plot import colors
from matplotlib.collections import PolyCollection

from citlab_article_separation.util import get_article_rectangles_from_baselines, merge_article_rectangles_vertically


def plot_gt_data(img_path, surr_polys_dict, show=True):
    """ Plots the groundtruth data for the article separation network or saves it to the directory

    :param img_path: path to the image the groundtruth is produced for
    :param article_rect_dict: the ArticleRectangle dictionary
    :param img_width: the width of the image
    :param img_height: the height of the image
    :param kernel_size: the size of the dilation kernel
    :param savedir: the directory the data should be stored
    :return:
    """
    fig, ax = plt.subplots()
    page_plot.add_image(ax, img_path)

    for i, a_id in enumerate(surr_polys_dict):
        # add facecolors="None" if rectangles should not be filled
        surr_polys = surr_polys_dict[a_id]
        if a_id == "blank":
            ar_poly_collection = PolyCollection(surr_polys, closed=True, edgecolors='k', facecolors='k')
        else:
            ar_poly_collection = PolyCollection(surr_polys, closed=True, edgecolors=colors.COLORS[i],
                                                facecolors=colors.COLORS[i])
        ar_poly_collection.set_alpha(0.5)
        ax.add_collection(ar_poly_collection)

    if show:
        plt.show()


def plot_polys_binary(polygon_list, img=None, img_width=None, img_height=None, closed=True, fill_articles=False):
    """Adds a list of polygons `polygon_list` to a pillow image `img`. If `img` is None a new pillow image is generated
    with a width of `img_width` and a height of `img_height`.

    :param polygon_list: a list of polygons, each given by a list of (x,y) tuples
    :type polygon_list: list of (list of (int, int))
    :param img: the pillow image to draw the polygons on
    :type img: Image.Image
    :param img_width: the width of the newly created image
    :type img_width: int
    :param img_height: the height of the newly created image
    :type img_height: int
    :param closed: draw a closed polygon or not
    :type closed: bool
    :return: pillow image
    """
    if img is None:
        # create binary image
        assert type(img_width) == int and type(img_height) == int, f"img_width and img_height must be integers but got " \
            f"the following values instead: {img_width} and {img_height}."
        img = Image.new('1', (img_width, img_height))
    pdraw = ImageDraw.Draw(img)
    for poly in polygon_list:
        if closed:
            if fill_articles:
                pdraw.polygon(poly, outline="white", fill="white")
            else:
                pdraw.polygon(poly, outline="white")
        else:
            pdraw.line(poly, fill="white", width=1)

    return img


def rescale_image(img=None):
    if img is None:
        print("You must provide an image in order to rescale it.")
        exit(1)
    pass


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def create_filenames_with_baseline_gt(save_folder, img_filename):
    article_gt_savefile = os.path.join(save_folder, "C3", img_filename + "_GT0.png")
    baseline_gt_savefile = os.path.join(args.save_folder, "C3", newspaper_filename + "_GT1.png")
    other_gt_savefile = os.path.join(args.save_folder, "C3", newspaper_filename + "_GT2.png")
    downscaled_grey_image_savefile = os.path.join(args.save_folder, newspaper_filename + ".png")
    rotation_savefile_name = downscaled_grey_image_savefile + ".rot"

    return article_gt_savefile, baseline_gt_savefile, other_gt_savefile, downscaled_grey_image_savefile, \
           rotation_savefile_name


def create_filenames_wo_baseline_gt(save_folder, img_filename):
    article_gt_savefile = os.path.join(save_folder, "C3", img_filename + "_GT0.png")
    other_gt_savefile = os.path.join(args.save_folder, "C3", newspaper_filename + "_GT1.png")
    downscaled_grey_image_savefile = os.path.join(args.save_folder, newspaper_filename + ".png")
    rotation_savefile = downscaled_grey_image_savefile + ".rot"

    return article_gt_savefile, other_gt_savefile, downscaled_grey_image_savefile, rotation_savefile


def create_filenames_ab_a(save_folder, img_filename):
    article_boundary_gt_savefile = os.path.join(save_folder, "C3", img_filename + "_GT0.png")
    article_gt_savefile = os.path.join(save_folder, "C3", img_filename + "_GT1.png")
    other_gt_savefile = os.path.join(args.save_folder, "C3", newspaper_filename + "_GT2.png")
    downscaled_grey_image_savefile = os.path.join(args.save_folder, newspaper_filename + ".png")
    rotation_savefile = downscaled_grey_image_savefile + ".rot"

    return article_gt_savefile, article_boundary_gt_savefile, other_gt_savefile, downscaled_grey_image_savefile, \
           rotation_savefile


def check_if_files_exist(*file_names):
    files_exist = map(os.path.isfile, file_names)
    return all(files_exist)


def convert_and_apply_dilation(img, mode='article', fill_articles=False):
    # other modes: baseline
    img_np = img.convert('L')
    img_np = np.array(img_np, np.uint8)

    if mode == 'article':
        if fill_articles:
            return img_np

        img_np = apply_transform(img_np, transform_type='dilation', kernel_size=(10, 10),
                                 kernel_type='rect',
                                 iterations=1)
        img_np = apply_transform(img_np, transform_type='erosion', kernel_size=(5, 5),
                                 kernel_type='rect',
                                 iterations=1)
    elif mode == 'baseline':
        img_np = apply_transform(img_np, transform_type='dilation', kernel_size=(1, 3),
                                 kernel_type='rect',
                                 iterations=1)

    return img_np


def create_baseline_gt_img(ar_dict, sc_factor, img_width, img_height):
    img_scaled_width = round(img_width * sc_factor)
    img_scaled_height = round(img_height * sc_factor)

    baseline_polygon_img = None
    for aid, ars in ar_dict.items():
        baseline_polygon_img = plot_polys_binary(
            [rescale_points(tl.baseline.points_list, sc_factor) for ar in ars for tl in
             ar.textlines],
            baseline_polygon_img, img_height=img_scaled_height, img_width=img_scaled_width, closed=False)

    baseline_polygon_img_np = convert_and_apply_dilation(baseline_polygon_img, mode='baseline')

    return baseline_polygon_img_np


def create_article_polygon_gt_img(surr_polys_dict, sc_factor, img_width, img_height, fill_articles):
    img_scaled_width = round(img_width * sc_factor)
    img_scaled_height = round(img_height * sc_factor)

    article_polygon_img = None
    for aid, surr_polys in surr_polys_dict.items():
        if aid is None:
            continue
        surr_polys_scaled = []
        for sp in surr_polys:
            sp_as_list = sp.as_list()
            surr_polys_scaled.append(rescale_points(sp_as_list, sc_factor))

        # returns a pillow image
        article_polygon_img = plot_polys_binary(surr_polys_scaled, article_polygon_img, img_height=img_scaled_height,
                                                img_width=img_scaled_width, fill_articles=fill_articles)

    article_polygon_img_np = convert_and_apply_dilation(article_polygon_img, mode='article',
                                                        fill_articles=fill_articles)

    return article_polygon_img_np


def create_other_gt_img(*channel_images):
    other_img_np = 255 * np.ones(channel_images[0].shape, np.uint8)

    for channel_img in channel_images:
        other_img_np -= channel_img

    other_img_np *= ((other_img_np == 0) + (other_img_np == 255))

    return other_img_np


def save_gt_data(savefile_name, img_np):
    cv2.imwrite(savefile_name, img_np)
    print(f'Saved file {savefile_name}')


def save_downscaled_grey_img(path_to_img, savefile_name, sc_factor, img_width_to_match, img_height_to_match):
    grey_img = Image.open(path_to_img).convert('L')
    assert grey_img.size == (img_width_to_match, img_height_to_match), f"resolutions of images don't match but are" \
        f"{grey_img.size} and ({img_width_to_match, img_height_to_match})"
    grey_img_np = cv2.resize(np.array(grey_img, np.uint8), None, fx=sc_factor, fy=sc_factor,
                             interpolation=cv2.INTER_AREA)
    cv2.imwrite(savefile_name, grey_img_np)
    print(f'Saved file {savefile_name}')


if __name__ == '__main__':
    jpype.startJVM(jpype.getDefaultJVMPath())
    parser = ArgumentParser()
    parser.add_argument('--path_to_xml_lst', default='', type=str,
                        help="path to the lst file containing the file paths of the PageXMLs.")
    parser.add_argument('--path_to_img_lst', default='', type=str,
                        help='path to the lst file containing the file paths of the images.')
    parser.add_argument('--scaling_factor', default=0.5, type=float,
                        help='how much the GT images will be down-sampled, defaults to 0.5.')
    parser.add_argument('--save_folder', default='', type=str,
                        help='path to the folder the GT is written to.')
    parser.add_argument('--fixed_img_height', default=0, type=int,
                        help='fix the height of the image to one specific value')
    parser.add_argument('--use_surr_polys', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to use the surrounding polygons of the baselines or not.')
    parser.add_argument('--use_stretch', type=str2bool, nargs='?', const=True, default=True,
                        help='whether to stretch the article rectangles to the top or not. Should be used if '
                             '"--use_surr_polys" is False.')
    parser.add_argument('--use_convex_hull', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to use the convex hull when merging article rectangles or ortho connect.')
    parser.add_argument('--use_max_rect_size', type=str2bool, nargs='?', const=True, default=False,
                        help='whether to use a maximal article rectangle size or not.')
    parser.add_argument('--min_width_intersect', default=10, type=int,
                        help='How much two article rectangles at least have to overlap '
                             'horizontally to connect them to one article rectangle in a postprocessing step.')
    parser.add_argument('--plot_page_xml', type=str2bool, nargs='?', const=True, default=True,
                        help='whether to plot_binary the PageXml or not.')
    parser.add_argument('--mode', default='ab_bl', type=str,
                        help='choose which GT you want to generate, choose from ["ab_bl", "ab", "a"].\n'
                             '\t ab_bl: article boundaries + baselines\n'
                             '\t ab: article boundaries\n'
                             '\t a: article filled')

    MODES = ['ab_bl', 'ab', 'a', 'ab_a']

    args = parser.parse_args()

    if args.mode.lower() not in MODES:
        raise ValueError(f'Please choose from one of the modes {MODES}.')

    if args.path_to_xml_lst == '':
        raise ValueError(f'Please provide a path to the list of PageXML files.')

    if args.path_to_img_lst == '':
        raise ValueError(f'Please provide a path to the list of image files.')

    if args.save_folder == '':
        raise ValueError(f'Please provide a valid save folder name.')

    if not os.path.exists(os.path.join(args.save_folder, 'C3')):
        os.makedirs(os.path.join(args.save_folder, 'C3'))

    with open(args.path_to_xml_lst) as f, open(args.path_to_img_lst) as g:
        for path_to_page_xml, path_to_img in zip(f.readlines(), g.readlines()):
            path_to_page_xml = path_to_page_xml.strip()
            path_to_img = path_to_img.strip()

            page_filename = os.path.basename(path_to_page_xml)
            newspaper_filename = os.path.splitext(page_filename)[0]

            if args.mode.lower() == "ab_bl":
                article_gt_filename, baseline_gt_filename, other_gt_filename, downscaled_grey_img_filename, rotation_filename = create_filenames_with_baseline_gt(
                    save_folder=args.save_folder, img_filename=newspaper_filename)
                files_exist = check_if_files_exist(article_gt_filename, baseline_gt_filename, other_gt_filename,
                                                   downscaled_grey_img_filename, rotation_filename)

            elif args.mode.lower() in ["a", "ab"]:
                article_gt_filename, other_gt_filename, downscaled_grey_img_filename, rotation_filename = create_filenames_wo_baseline_gt(
                    save_folder=args.save_folder, img_filename=newspaper_filename)
                files_exist = check_if_files_exist(article_gt_filename, other_gt_filename, downscaled_grey_img_filename,
                                                   rotation_filename)

            elif args.mode.lower() == "ab_a":
                article_gt_filename, article_boundary_gt_filename, other_gt_filename, downscaled_grey_img_filename, rotation_filename = create_filenames_ab_a(
                    save_folder=args.save_folder, img_filename=newspaper_filename)
                files_exist = check_if_files_exist(article_gt_filename, article_boundary_gt_filename, other_gt_filename,
                                                   downscaled_grey_img_filename, rotation_filename)

            if files_exist:
                print(
                    f"GT Files for PageXml {path_to_page_xml} already exist, skipping...")
                continue

            # TODO: only generates files with '0's in it -> fix this
            with open(rotation_filename, "w") as rot:
                rot.write("0")

            page = Page(path_to_page_xml)
            img_width, img_height = page.get_image_resolution()
            article_rectangle_dict = get_article_rectangles_from_baselines(page, path_to_img,
                                                                           use_surr_polygons=args.use_surr_polys,
                                                                           stretch=args.use_stretch)

            if args.fixed_img_height:
                sc_factor = args.fixed_img_height / img_height
            else:
                sc_factor = args.scaling_factor

            surr_polys_dict = merge_article_rectangles_vertically(article_rectangle_dict,
                                                                  min_width_intersect=args.min_width_intersect,
                                                                  use_convex_hull=args.use_convex_hull)

            if args.plot_page_xml:
                page_plot.plot_pagexml(page, path_to_img)
            plot_gt_data(path_to_img,
                         {aid: [poly.as_list() for poly in poly_list] for aid, poly_list in surr_polys_dict.items() if
                          aid is not None})

            if args.mode == "ab_bl":
                article_polygon_img_np = create_article_polygon_gt_img(surr_polys_dict, sc_factor, img_width,
                                                                       img_height, fill_articles=False)
                baseline_polygon_img_np = create_baseline_gt_img(article_rectangle_dict, sc_factor, img_width,
                                                                 img_height)
                save_gt_data(baseline_gt_filename, baseline_polygon_img_np)

                other_img_np = create_other_gt_img(article_polygon_img_np, baseline_polygon_img_np)
            elif args.mode == "ab":
                article_polygon_img_np = create_article_polygon_gt_img(surr_polys_dict, sc_factor, img_width,
                                                                       img_height, fill_articles=False)
                other_img_np = create_other_gt_img(article_polygon_img_np)
            elif args.mode == "a":
                article_polygon_img_np = create_article_polygon_gt_img(surr_polys_dict, sc_factor, img_width,
                                                                       img_height, fill_articles=True)
                article_polygon_bounds_img_np = create_article_polygon_gt_img(surr_polys_dict, sc_factor, img_width,
                                                                              img_height, fill_articles=False)
                article_polygon_img_np -= article_polygon_bounds_img_np
                other_img_np = create_other_gt_img(article_polygon_img_np)
            elif args.mode == "ab_a":
                article_polygon_img_np = create_article_polygon_gt_img(surr_polys_dict, sc_factor, img_width,
                                                                       img_height, fill_articles=True)
                article_polygon_bounds_img_np = create_article_polygon_gt_img(surr_polys_dict, sc_factor, img_width,
                                                                              img_height, fill_articles=False)
                article_polygon_img_np -= article_polygon_bounds_img_np
                save_gt_data(article_boundary_gt_filename, article_polygon_bounds_img_np)
                other_img_np = create_other_gt_img(article_polygon_img_np, article_polygon_bounds_img_np)

            save_gt_data(article_gt_filename, article_polygon_img_np)
            save_gt_data(other_gt_filename, other_img_np)
            save_downscaled_grey_img(path_to_img, downscaled_grey_img_filename, sc_factor, img_width, img_height)

    jpype.shutdownJVM()
