import logging
import os
from abc import ABC, abstractmethod
from collections import namedtuple, defaultdict
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw
from python_util.geometry.point import rescale_points
from python_util.io.file_loader import load_list_file
from python_util.io.path_util import get_page_from_img_path
from python_util.parser.xml.page.page import Page

logger = logging.getLogger("GroundTruthGenerator")


class GroundTruthGenerator(ABC):
    """
    Abstract base class for different kinds of ground truth generators for image segmentation / pixel labelling tasks.
    """
    def __init__(self, path_to_img_lst, max_resolution=(0, 0), scaling_factor=1.0):
        self.img_path_lst = load_list_file(path_to_img_lst)
        self.valid_img_indizes = []
        self.page_path_lst = [get_page_from_img_path(img_path) for img_path in self.img_path_lst]
        self.page_object_lst = self.create_page_objects()
        # list of tuples (width, height) containing the size of the image
        self.img_res_lst_original = self.get_image_resolutions_from_page_objects()
        self.max_resolution = (max(0, max_resolution[0]), max(0, max_resolution[1]))
        if self.max_resolution is not (0, 0):
            self.scaling_factors = self.calculate_scaling_factors_from_max_resolution()
        else:
            sc_factor = max(0.1, scaling_factor)
            self.scaling_factors = [sc_factor] * len(self.img_path_lst)
        self.images_list, self.img_res_lst = self.create_images()
        self.gt_imgs_lst = []  # list of tuples
        self.gt_polygon_lst = []  # list of tuples representing lists of polygons plotted in gt_imgs_lst
        self.n_channels = 0

        self.regions_dict = {}
        self.gt_dict = defaultdict(list)
        self.regions_information_dict = {}
        self.RegionInfo = namedtuple('RegionInfo', ['num_regions', 'pixel_percentages'])

        super().__init__()

    def create_page_objects(self):
        return [Page(page_path) for page_path in self.page_path_lst]

    def create_images(self, color_mode='L'):
        """
        Create (scaled) grey value versions of the original input images and return them in a list together with the new
        image resolutions
        :param color_mode: Which color mode to use, defaults to 'L' for grey value images.
        :return: List of the (scaled) grey value images and a list of the corresponding new image resolutions.
        """
        new_img_res_lst = []
        grey_img_lst = []
        for i, path_to_img in enumerate(self.img_path_lst):
            grey_img = Image.open(path_to_img).convert(color_mode)
            grey_img = np.array(grey_img, np.uint8)

            if self.scaling_factors[i] < 1:
                grey_img = cv2.resize(grey_img, None, fx=self.scaling_factors[i], fy=self.scaling_factors[i],
                                      interpolation=cv2.INTER_AREA)
            elif self.scaling_factors[i] > 1:
                grey_img = cv2.resize(grey_img, None, fx=self.scaling_factors[i], fy=self.scaling_factors[i],
                                      interpolation=cv2.INTER_CUBIC)

            new_img_res_lst.append(grey_img.shape)
            grey_img_lst.append(grey_img)

        return grey_img_lst, new_img_res_lst

    @abstractmethod
    def create_ground_truth_images(self):
        pass

    def create_and_write_info_file(self, path_to_info_file):
        """
        Creates an info file holding information about the ground truth, e.g. which channel belongs to which region. The
        info file is then stored in ```path_to_info_file```.
        :param path_to_info_file: Save path of the info file.
        :return:
        """
        with open(path_to_info_file, "w") as info_file:
            info_file.write(f"Processed {len(self.img_path_lst)} images.\n\n")
            info_file.write("GT channels:\n")
            for i, region_name in enumerate(self.regions_dict.keys()):
                info_file.write(f"\tGT{i}: {region_name}\n")
            info_file.write("\n")

            for region_name, region_info in self.regions_information_dict.items():
                num_images = len(region_info[0]) - region_info[0].count(0)
                num_regions_overall = sum(region_info[0])
                avg_pixel_percentage = np.average(region_info[1])
                info_file.write(region_name)
                info_file.write(f"\tNumber of images: {num_images}\n")
                info_file.write(f"\tNumber of regions overall: {num_regions_overall}\n")
                info_file.write(f"\tAverage pixel percentage: {avg_pixel_percentage}\n")

    def add_region_information(self):
        """
        For each region in the ground truth files the number of regions and the total amount of ground truth pixels is
        stored inside a dictionary.
        :return: Dictionary with region information.
        """
        for region_name, region_list in self.regions_dict.items():
            self.regions_information_dict[region_name] = self.RegionInfo([len(region) for region in region_list],
                                                                         [np.count_nonzero(gt) / gt.size
                                                                          for gt in self.gt_dict[region_name]])

    def save_ground_truth(self, save_dir):
        """
        Save the ground images together with the (scaled) grey value images and the rotation files inside ```save_dir```.
        :param save_dir: Path to the folder where all related ground truth files should be stored.
        :return:
        """
        if not self.gt_imgs_lst:
            print("No ground truth images to save.")
        else:
            if not os.path.isdir(save_dir) or not os.path.exists(save_dir):
                os.mkdir(save_dir)

            for i, gt_imgs in enumerate(self.gt_imgs_lst):
                for j, gt_img in enumerate(gt_imgs):
                    gt_img_savefile_name = self.get_ground_truth_image_savefile_name(
                        self.img_path_lst[self.valid_img_indizes[i]], j, save_dir,
                        gt_folder_name="C" + str(len(gt_imgs)))
                    cv2.imwrite(gt_img_savefile_name, gt_img)
                cv2.imwrite(self.get_grey_image_savefile_name(self.img_path_lst[self.valid_img_indizes[i]], save_dir),
                            self.images_list[self.valid_img_indizes[i]])
                with open(self.get_rotation_savefile_name(self.img_path_lst[self.valid_img_indizes[i]], save_dir), "w") as rot:
                    rot.write("0")

    @staticmethod
    def create_other_ground_truth_image(*channel_images):
        """
        Takes as input all ground truth channels and creates the "other" ground truth channel by subtracting a binary
        image with every pixel set to "on" (e.g. "on"=255 and "off"=0) with the union of all ground truth channels.
        :param channel_images: All ground truth images stored in a list.
        :return: Ground truth channel "other".
        """
        other_img_np = 255 * np.ones(channel_images[0].shape, np.uint8)

        for channel_img in channel_images:
            other_img_np -= channel_img

        other_img_np *= ((other_img_np == 0) + (other_img_np == 255))

        return other_img_np

    @staticmethod
    def get_ground_truth_image_savefile_name(img_name, index, save_dir, gt_folder_name="C3", gt_file_ext=".png"):
        """
        Returns the save paths of the created ground truth images (one for each class/channel). For
        `img_name="/path/to/image.jpg"`, `index=0`, `save_dir="/path/to/folder"`, `gt_folder_name="C3"` and
        `gt_file_ext=".png"` the function would return `"/path/to/folder/C3/image_GT0.png"`.
        :param img_name: Path to the original image for which the ground truth is created.
        :param index: Index of the ground truth channel.
        :param save_dir: Path to the folder where the folder with all ground truth images should be stored.
        :param gt_folder_name: Name of the folder where all ground truth files should be stored.
        :param gt_file_ext: File extension of the ground truth images.
        :return: Save path of a ground truth image with a specific index.
        """
        channel_gt_dir = os.path.join(save_dir, gt_folder_name)
        if not os.path.exists(channel_gt_dir) or not os.path.isdir(channel_gt_dir):
            os.mkdir(channel_gt_dir)
        img_name_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
        return os.path.join(save_dir, gt_folder_name, img_name_wo_ext + "_GT" + str(index) + gt_file_ext)

    @staticmethod
    def get_grey_image_savefile_name(img_name, save_dir, grey_img_file_ext=".jpg"):
        """
        Returns the save path of the created grey value image that is a grey valued and (perhaps) scaled version of the
        original input. This file is stored inside ```save_dir``` and has a default file extension of `.jpg`.

        :param img_name: Path to the original image for which the ground truth is created.
        :param save_dir: Path to the folder where the (scaled) grey value image should be stored.
        :param grey_img_file_ext: File extension of the grey value image.
        :return: Save path of the grey value image.
        """
        img_name_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
        return os.path.join(save_dir, img_name_wo_ext + grey_img_file_ext)

    @staticmethod
    def get_rotation_savefile_name(img_name, save_dir, rotation_file_ext=".jpg.rot"):
        """
        Returns the save path of the `.rot` file that holds information about the rotation of the image. This file
        is stored inside ```save_dir``` and has a default file extension of `.jpg.rot`.
        :param img_name: Path to the original image for which the ground truth is created.
        :param save_dir: Path to the folder where the rotation file should be stored.
        :param rotation_file_ext: File extension of the rotation file.
        :return: Save path of the rotation file.
        """
        img_name_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
        return os.path.join(save_dir, img_name_wo_ext + rotation_file_ext)

    def run_ground_truth_generation(self, save_dir, create_info_file=True):
        """
        Runs the actual ground truth generation process and stores the results inside the folder ```save_dir```. If
        ```create_info_file``` is True, an additional file with information regarding the channels is created inside
        ```save_dir```.
        :param save_dir: Path to the folder where the binary ground images are saved.
        :param create_info_file: If True, an additional info file regarding the channels is created inside the save
        directory.
        :return:
        """
        self.create_images()
        if len(self.page_object_lst) < 1:
            self.create_page_objects()
        self.create_ground_truth_images()

        if create_info_file:
            self.add_region_information()
            self.create_and_write_info_file(os.path.join(save_dir, 'info.txt'))

        self.save_ground_truth(save_dir)

    @staticmethod
    def rescale_polygon(polygon, scaling_factor):
        """
        Rescale the polygon ```polygon``` according to the scaling factor ```scaling_factor```.
        :param polygon: Polygon given as a list of (x,y) coordinates.
        :param scaling_factor: Scaling factor that should be applied to the points of the given polygon.
        :return: A scaled version of the given polygon.
        """
        return rescale_points(polygon, scaling_factor) if scaling_factor else polygon

    @staticmethod
    def plot_polys_binary(polygon_list, img=None, img_width=None, img_height=None, closed=True, fill_polygons=False,
                          line_width=7):
        """
        Plot the polygons in ```polygon_list``` to a numpy array. If an image ```img``` is already given, add the polygons.
        If no image is given, values for the width (```img_width```) and height (```img_height```) of the image must be provided.
        If ```closed``` is True, a line is drawn from the last point of the polygon to the first one. If ```fill_polygons``` is
        True (provided ```closed``` is also True), the polygons are filled, otherwise only the border is drawn. Finally,
        ```line_width``` is controlling the thickness of the polygonal chain.
        :param polygon_list: A list of polygons given by (x,y) coordinates.
        :param img: A numpy array where the polygons should be added. If None a new one is generated.
        :param img_width: The width of the newly created image if an image was not already given.
        :param img_height: The height of the newly created image if an image was not already given.
        :param closed: Determines if the last point of the polygon is connected to its first.
        :param fill_polygons: Determines if the polygons should be filled when drawn.
        :param line_width: Controls the thickness of the polygonal chain.
        :return: numpy array with polygons added either to a given image or to a newly created one.
        """
        if img is None:
            # create binary image
            assert type(img_width) == int and type(
                img_height) == int, f"img_width and img_height must be integers but got " \
                                    f"the following values instead: {img_width} and {img_height}."
            img = Image.new('1', (img_width, img_height))
        pdraw = ImageDraw.Draw(img)
        for poly in polygon_list:
            if closed:
                if fill_polygons:
                    pdraw.polygon(poly, outline="white", fill="white")
                else:
                    poly_closed = deepcopy(poly)
                    poly_closed.append(poly[0])
                    # pdraw.polygon(poly, outline="white")
                    pdraw.line(poly_closed, fill="white", width=line_width)
            else:
                pdraw.line(poly, fill="white", width=line_width)

        img = img.convert('L')
        return np.array(img, np.uint8)

    @staticmethod
    def make_disjoint(gt_img_compare, gt_img_to_change):
        """
        Make two binary images disjoint by subtracting ```gt_image_compare``` from ```gt_img_to_change```. If the size of both
        images don't match an exception is raised.
        :param gt_img_compare: binary image to subtract
        :param gt_img_to_change: binary image to subtract from
        :return: ```gt_img_to_change``` - ```gt_img_compare```
        """
        # TODO: Check if dimensions of both images match (assert ...)
        return np.where(gt_img_compare == gt_img_to_change, 0, gt_img_to_change)

    def make_disjoint_all(self):
        """
        Make the binary ground truth channels stored in `self.gt_imgs_lst` disjoint, where the first ground truth
        channel has the highest priority and the last one the least priority. I.e., the first channel stays as it is
        and for each following channel the bitwise OR of all previous channels is subtracted.
        :return:
        """
        for i, gt_imgs in enumerate(self.gt_imgs_lst):
            gt_img_compare = gt_imgs[0]
            changed_gt_imgs = [gt_imgs[0]]
            for j in range(len(gt_imgs) - 1):
                changed_gt_imgs.append(
                    self.make_disjoint(gt_img_compare=gt_img_compare, gt_img_to_change=gt_imgs[j + 1]))
                gt_img_compare = np.bitwise_or(gt_img_compare, gt_imgs[j + 1])

            self.gt_imgs_lst[i] = tuple(changed_gt_imgs)

    def get_image_resolutions_from_page_objects(self):
        return [page.get_image_resolution() for page in self.page_object_lst]

    def calculate_scaling_factors_from_max_resolution(self):
        """
        Given the ```max_resolution``` parameter return the corresponding scaling factors for each image.
        A distinction is made between the following cases, where ``max_resolution`` is
            - (0, 0): return scaling factor 1 for each image
            - (0, y): return scaling factors in (0, 1], s.t. the final image height is <= y
            - (x, 0): return scaling factors in (0, 1], s.t. the final image width is <= x
            - (x, y): return scaling factors in (0, 1], s.t. the final image height and image width are <= (x, y)
        :return:
        """
        if self.max_resolution == (0, 0):
            logger.debug("No max resolution given, do nothing...")
            return [1.0] * len(self.img_res_lst_original)

        # if image height is too big, scale down
        if self.max_resolution[0] == 0:
            return [min(1.0, self.max_resolution[1] / img_res[0]) for img_res in self.img_res_lst_original]
        # if image width is too big, scale down
        elif self.max_resolution[1] == 0:
            return [min(1.0, self.max_resolution[0] / img_res[1]) for img_res in self.img_res_lst_original]
        # if image height and/or image_width are too big, scale down
        else:
            return [min(1.0, max(self.max_resolution[1] / img_res[0], self.max_resolution[0] / img_res[1])) for img_res
                    in
                    self.img_res_lst_original]
