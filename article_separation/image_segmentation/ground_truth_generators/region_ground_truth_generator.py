import argparse
import logging
import json
import os

import cv2
import numpy as np
from python_util.parser.xml.page import page_constants
from python_util.io.path_util import prepend_folder_name
from python_util.basic.flags import str2bool

from article_separation.image_segmentation.ground_truth_generators.ground_truth_generator_base import \
    GroundTruthGenerator

from tqdm import tqdm

from typing import List, Tuple

logger = logging.getLogger("TextBlockGroundTruthGenerator")
logging.basicConfig(level=logging.WARNING)


class RegionGroundTruthGenerator(GroundTruthGenerator):
    """
    A ground truth generator specific for regions that can be extracted from the metadata inside PAGE-XML files.
    """
    def __init__(self, path_to_img_lst, max_resolution=(0, 0), scaling_factor=1.0, use_bounding_box=False,
                 use_min_area_rect=False):
        super().__init__(path_to_img_lst, max_resolution, scaling_factor)
        self.regions_list = [page.get_regions() for page in self.page_object_lst]
        self.image_regions_list = self.get_image_regions_list()
        self.separator_regions_list = self.get_separator_regions_list()
        # self.text_regions_list = self.get_valid_text_regions()
        self.text_regions_list = self.get_valid_text_regions(intersection_thresh=-1,
                                                             region_types=[page_constants.TextRegionTypes.sPARAGRAPH,
                                                                           page_constants.TextRegionTypes.sHEADING])
        self.heading_regions_list = self.get_valid_text_regions(intersection_thresh=-1,
                                                                region_types=[page_constants.TextRegionTypes.sHEADING])
        self.use_bounding_box = use_bounding_box
        self.use_min_area_rect = use_min_area_rect
        # TODO: Make this a class variable?
        if args.save_json:
            self.image_list, self.img_res_lst = super().create_images(color_mode='RGB')

    def run_ground_truth_generation(self, save_dir, create_info_file=True):
        """
        Runs the ground truth generation as defined in ```create_ground_truth_json``` and otherwise as defined in the super
        class. The files are saved to ```save_dir``` with an optional info file (only for the non-json case) based on the
        ```create_info_file``` parameter.
        :param save_dir: Path to the save folder.
        :param create_info_file: Optionally create an info file.
        :return:
        """
        if args.save_json:
            self.scaling_factors = [1] * len(self.img_path_lst)
            if len(self.page_object_lst) < 1:
                self.create_page_objects()
            self.create_ground_truth_json(save_dir, self.text_regions_list, enforce_unique_name=False)
        else:
            super().run_ground_truth_generation(save_dir, create_info_file)

    def create_ground_truth_json(self, save_folder, regions_list=None, enforce_unique_name=False):
        """
        Creates a json file containing the region information (only text regions for now!) for each image in the
        following format:
         { 'image_name1.jpg':
           { 'regions': {
               '0': {
                    'x_points': [...],
                    'y_points': [...],
                    'class_name': 'textblock'
                    },
               ... more regions ...
               },
             'height': 1500,
             'width': 800
           },
           'image_name2.jpg':
           { 'regions': {
               ...
               }
             'height': 1600,
             'width': 900
           },
           ... more images ...
         }

         A json style format is needed for training the MASK-RCNN model as defined in
         https://github.com/matterport/Mask_RCNN.
        :param save_folder: Path to the save folder.
        :type save_folder: str
        :param regions_list: A list of regions for each image. If None, the list of regions is created.
        :type regions_list: List[List[List[Tuple[int, int]]]]
        :param enforce_unique_name: Enforce unique filenames by prepending the folder names of the original input
        images, since the files are stored in a flat hierarchy.
        :type enforce_unique_name: bool
        :return:
        """
        if regions_list is None:
            regions_list = self.text_regions_list

        data = {}
        for i in tqdm(range(len(self.img_path_lst))):
            if enforce_unique_name:
                image_path_new = prepend_folder_name(self.img_path_lst)
                image_name = os.path.basename(image_path_new)
            else:
                image_name = os.path.basename(self.img_path_lst[i])

            img_height = self.img_res_lst[i][0]
            img_width = self.img_res_lst[i][1]

            regions = regions_list[i]
            regions_dict = {}
            for j, region in enumerate(regions):
                region_polygon = region.points.to_polygon()
                x_points = region_polygon.x_points
                y_points = region_polygon.y_points
                if x_points[0] != x_points[-1] or y_points[0] != y_points[-1]:
                    x_points.append(x_points[0])
                    y_points.append(y_points[0])
                class_name = "textblock"
                regions_dict[str(j)] = {'x_points': x_points, 'y_points': y_points, 'class_name': class_name}

            if image_name in data.keys():
                raise Exception("Key already existent, please try to prepend the folder name to the file name.")

            data[image_name] = {'regions': regions_dict, 'height': img_height, 'width': img_width}

            image_save_path = os.path.join(save_folder, image_name)
            if os.path.exists(image_save_path):
                continue
            os.symlink(src=self.img_path_lst[i], dst=image_save_path)

        json_save_file_name = os.path.join(save_folder, 'regions.json')
        with open(json_save_file_name, 'w') as json_file:
            json.dump(data, json_file)

    # TODO: Make this parameterizable, by e.g. adding flags for each region type.
    def create_ground_truth_images(self):
        """
        Create the ground truth images based on what regions should be recognized later on.
        :return:
        """
        # Textblocks outlines, Textblocks filled, (Images), Separators, Other
        for i in range(len(self.img_path_lst)):
            # Textblock outline GT image
            img_width = self.img_res_lst[i][1]
            img_height = self.img_res_lst[i][0]
            sc_factor = self.scaling_factors[i]

            # tb_outlines_gt_img = self.create_region_gt_img(self.text_regions_list[i], img_width, img_height, fill=False,
            #                                                scaling_factor=sc_factor)
            tb_filled_gt_img = self.create_region_gt_img(self.text_regions_list[i], img_width, img_height, fill=True,
                                                         scaling_factor=sc_factor)
            # image_region_gt_img = self.create_region_gt_img(self.image_regions_list[i], img_width, img_height, fill=True,
            #                                                 scaling_factor=sc_factor)
            # sep_region_gt_img = self.create_region_gt_img(self.separator_regions_list[i], img_width, img_height,
            #                                               fill=True, scaling_factor=sc_factor)
            gt_channels = [tb_filled_gt_img]
            # gt_channels = [tb_outlines_gt_img, tb_filled_gt_img, sep_region_gt_img]
            # gt_channels = [tb_outlines_gt_img, tb_filled_gt_img, image_region_gt_img, sep_region_gt_img]

            other_gt_img = self.create_other_ground_truth_image(*gt_channels)
            gt_channels.append(other_gt_img)
            gt_channels = tuple(gt_channels)

            self.gt_imgs_lst.append(gt_channels)
            self.valid_img_indizes.append(i)
        self.make_disjoint_all()

    # TODO: Move this to python_util?
    @staticmethod
    def get_min_area_rect(points):
        """
        Returns the minimal enclosing aligned rectangle of a set of points given by ```points```.
        :param points: List of (x,y) coordinates.
        :type points: List[Tuple[int, int]]
        :return: Minimal enclosing aligned rectangle of a set of points.
        """
        points_np = np.array(points)
        min_area_rect = cv2.minAreaRect(points_np)
        min_area_rect = cv2.boxPoints(min_area_rect)
        min_area_rect = np.int0(min_area_rect)

        min_area_rect = min_area_rect.tolist()
        min_area_rect = [tuple(p) for p in min_area_rect]

        return min_area_rect

    def create_region_gt_img(self, regions, img_width: int, img_height: int, fill: bool, scaling_factor: float = None):
        """
        Creates the region ground truth images for all regions in ```regions``` according to the function
        ```plot_polys_binary```. If ```scaling_factor``` is not ```None``` the polygons get scaled according to this value.
        :param regions: List of regions.
        :type regions: List[List[Tuple[int, int]]]
        :param img_width: Width of the final ground truth image.
        :type img_width: int
        :param img_height: Height of the final ground truth image.
        :type img_width: int
        :param fill: Determines if the regions should be filled or not.
        :type fill: bool
        :param scaling_factor: Scaling factor for the polygons.
        :type scaling_factor: float
        :return: One region ground truth image for each entry in the regions list.
        """
        if self.use_bounding_box:
            regions_polygons = [region.points.to_polygon().get_bounding_box().get_vertices() for region in regions]
        elif self.use_min_area_rect:
            regions_polygons = [self.get_min_area_rect(region.points.to_polygon().as_list()) for region in regions]
        else:
            regions_polygons = [region.points.to_polygon().as_list() for region in regions]

        region_gt_img = self.plot_polys_binary([self.rescale_polygon(rp, scaling_factor) for rp in regions_polygons],
                                               img_width=img_width, img_height=img_height, fill_polygons=fill,
                                               closed=True)
        return region_gt_img

    def get_valid_text_regions(self, intersection_thresh=20, region_types=None):
        """
        Get valid TextRegions (according to ```region_types```) from the PAGE file, where we check for intersections with images.
        If ```intersection_thresh``` is non-negative, allow intersections of text regions with images of up to this value,
        otherwise reject it. The intersection is the number of pixels the bounding boxes of the region and an image have
        in common. If ```intersection_thresh``` is negative, ignore the intersection and return all text regions
        of the types given in ```region_types```.
        :param intersection_thresh: An integer value that controls if a text region is rejected or not, depending on the
        intersection of the text region with an image region.
        :type intersection_thresh: int
        :param region_types: Valid region types for the final text regions, e.g. "TextRegion".
        :return:
        """
        if region_types is None:
            region_types = [page_constants.TextRegionTypes.sPARAGRAPH]
        if intersection_thresh < 0:
            return [[region for region in regions[page_constants.sTEXTREGION] if region.region_type in region_types]
                    for regions in self.regions_list]

        valid_text_regions_list = []
        for i, regions in enumerate(self.regions_list):
            valid_text_regions = []
            text_regions = [region for region in regions["TextRegion"] if region.region_type in region_types]
            image_regions = self.image_regions_list[i]
            if not image_regions:
                valid_text_regions_list.append(text_regions)
                continue

            text_regions_bbs = [text_region.points.to_polygon().get_bounding_box() for text_region in text_regions]
            image_regions_bbs = [image_region.points.to_polygon().get_bounding_box() for image_region in image_regions]

            for j, text_region_bb in enumerate(text_regions_bbs):
                for image_region_bb in image_regions_bbs:
                    if image_region_bb.contains_rectangle(text_region_bb):
                        break
                    intersection = text_region_bb.intersection(image_region_bb)
                    if intersection.height > intersection_thresh and intersection.width > intersection_thresh:
                        break
                else:
                    valid_text_regions.append(text_regions[j])

            valid_text_regions_list.append(valid_text_regions)

        return valid_text_regions_list

    def get_table_regions_list(self):
        """
        Returns all table regions from the created regions list.
        :return: List of table regions
        """
        return self.get_regions_list([page_constants.sTABLEREGION])

    def get_advert_regions_list(self):
        """
        Returns all advertisement regions from the created regions list.
        :return: List of advertisement regions.
        """
        return self.get_regions_list([page_constants.sADVERTREGION])

    def get_image_regions_list(self):
        """
        Returns all image regions from the created regions list.
        :return: List of image regions.
        """
        return self.get_regions_list([page_constants.sGRAPHICREGION, page_constants.sIMAGEREGION])

    def get_separator_regions_list(self):
        """
        Returns all separator regions from the created regions list.
        :return: List of separator regions.
        """
        return self.get_regions_list([page_constants.sSEPARATORREGION])

    def get_regions_list(self, region_types):
        """
        Returns all regions from the created regions list according to the region types given by ```region_types```.
        :param region_types: List of region types that should be returned.
        :type region_types: List[str]
        :return: All regions from the created regions list according to the given region types.
        """
        region_list_by_type = []
        for i, page_regions in enumerate(self.regions_list):
            regions = []
            for region_type in region_types:
                try:
                    regions += page_regions[region_type]
                except KeyError:
                    logger.debug("No {} for PAGE {}.".format(region_type, self.page_path_lst[i]))
                region_list_by_type.append(regions)

        return region_list_by_type

    def get_title_regions_list(self, title_region_types):
        """
        Returns all title regions from the created regions list.

        Valid title_region_types are ["headline", "subheadline", "publishing_stmt", "motto", "other" ]
        :param title_region_types: List of title region types that should be extracted.
        :type title_region_types: List[str]
        :return: List of title regions.
        """

        return self.get_heading_regions_list('title', title_region_types)

    def get_classic_heading_regions_list(self, heading_region_types):
        """
        Returns all "classic" heading regions from the created regions list.

        Valid "classic" heading region types are ["overline", "", "subheadline", "author", "other"]
        where "" represents the title (can also be "title").
        :param heading_region_types: List of heading regions types that should be extracted.
        :return: List of heading regions.
        """
        return self.get_heading_regions_list('heading', heading_region_types)

    def get_caption_text_regions(self):
        """
        Returns all caption regions from the created regions list.
        :return: List of caption regions.
        """
        return self.get_valid_text_regions(region_types=[page_constants.TextRegionTypes.sCAPTION])

    def get_heading_regions_list(self, custom_structure_type, custom_structure_subtypes):
        """
        Return all heading regions from the created regions list according to the types defined in the custom tag of a
        TextRegion.
        :param custom_structure_type: The value of the structure_type entry of the custom tag, e.g. "heading".
        :param custom_structure_subtypes: The value of the structure_subtype entry of the custom tag, e.g. "headline".
        :return: List of heading regions with a specific structure type and subtype.
        """
        valid_text_regions = self.get_valid_text_regions(region_types=[page_constants.TextRegionTypes.sHEADING])

        region_list_by_type = []
        if len(valid_text_regions) == 0:
            region_list_by_type.append([])
        for page_text_regions in valid_text_regions:
            regions = []
            for page_text_region in page_text_regions:
                custom_dict_struct = page_text_region.custom['structure']
                for custom_struct_subtype in custom_structure_subtypes:
                    if custom_struct_subtype == '' and custom_dict_struct[
                        'type'] == custom_structure_type and 'subtype' not in custom_dict_struct.keys():
                        regions.append(page_text_region)
                    elif custom_dict_struct['type'] == custom_structure_type and custom_dict_struct[
                        'subtype'] == custom_struct_subtype:
                        regions.append(page_text_region)
            region_list_by_type.append(regions)

        return region_list_by_type


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--max_height', type=int, default=0)
    parser.add_argument('--max_width', type=int, default=0)
    parser.add_argument('--scaling_factor', type=float, default=1.0)
    parser.add_argument('--save_json', nargs='?', const=True, default=True, type=str2bool,
                        help="If true, a json file for all images in the image list is created and saved into the"
                             "directory given by 'save_dir'. Also, symlinks to the original image files are created.")

    args = parser.parse_args()

    tb_generator = RegionGroundTruthGenerator(
        args.image_list, use_bounding_box=False, use_min_area_rect=False,
        max_resolution=(args.max_height, args.max_width), scaling_factor=args.scaling_factor)
    # print(tb_generator.image_regions_list)
    # print(tb_generator.text_regions_list)
    # tb_generator.create_images()
    # tb_generator.create_page_objects()
    # img_height = tb_generator.img_res_lst[0][0]
    # img_width = tb_generator.img_res_lst[0][1]
    # tb_gt = tb_generator.create_region_gt_img(tb_generator.text_regions_list[0], img_width, img_height, fill=True)
    # tb_surr_poly_gt = tb_generator.create_region_gt_img(tb_generator.text_regions_list[0], img_width, img_height,
    #                                                     fill=False)
    # cv2.imwrite("./data/tb_gt.png", tb_gt)
    # cv2.imwrite("./data/tb_surr_poly_gt.png", tb_surr_poly_gt)

    # imgplot = plt.imshow(gt_img, cmap="gray")
    # plt.show()
    # cv2.imshow("test", gt_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    tb_generator.run_ground_truth_generation(args.save_dir)
