import logging
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.image_processing.image_stats import get_rotation_angle
from scipy.ndimage import interpolation as inter

logger = logging.getLogger("TextBlockNetPostProcessor")
# logging.basicConfig(level=logging.WARNING)
logging.basicConfig(level=logging.INFO)

MIN_PIXEL_SEPARATOR_DISTANCE_FACTOR = 0.003
MAX_RECURSION_DEPTH = 4


class TextBlockNetPostProcessor(object):
    """Comments / Workflow:
        1.) the original image is used to calculate the rotation angle of the image -> better way to do this?
        2.) the text block channel of the net output is used to calculate white runs in the image, i.e. separator
        3.) the separator channel of the net output is used to extract visible separator from the image
        4.) 2.) & 3.) are combined to provide a first partition into coarse regions (the number of columns should be visible)
        5.) Iterate over the regions from the last step and use the separator and text block channel to provide more horizontal separator
        6.) The resulting grid-like image can be used to divide the Page into text regions

        won't work well for pages with
            - images, since no image detection is provided for now -> coming
            - complex layout, e.g. many advertisments -> check

    """

    def __init__(self, original_image, text_block_outline, text_block, separator):
        self.images = {'original_image': original_image, 'text_block_outline': text_block_outline,
                       'text_block': text_block, 'separator': separator,
                       'binarized_image': self.binarize_image(original_image),
                       'empty_image': np.zeros(original_image.shape, dtype=np.uint8)}
        if not self.check_dimensions(*self.images.values()):
            raise RuntimeError("Image shapes don't match.")
        self.image_height, self.image_width = self.images['original_image'].shape

    @staticmethod
    def binarize_net_output(image, threshold):
        return np.array((image > threshold), np.int32)

    @staticmethod
    def binarize_image(image, gaussian_blur=True):
        if gaussian_blur:
            res = cv2.GaussianBlur(image, (5, 5), 0)
        else:
            res = image
        _, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return res

    def get_best_rotation_angle(self):
        rotation_angle_binarized_image = get_rotation_angle(self.images['binarized_image'])[1]
        rotation_angle_textblock_image = get_rotation_angle(self.images["text_block"])[1]
        print(f"Rotation angle determined by the binarized image: {rotation_angle_binarized_image}")
        print(f"Rotation angle determined by the text block image: {rotation_angle_textblock_image}")
        return rotation_angle_binarized_image
        # return get_rotation_angle(self.images['binarized_image'])[1]

    @staticmethod
    def check_dimensions(*images):
        return all(image.shape == images[0].shape for image in images)

    def rotate_images(self, angle):
        logger.info(f"Rotate images by {angle} degrees.")
        for img_name, img in self.images.items():
            self.images[img_name] = inter.rotate(img, angle, reshape=False, order=0)

    @staticmethod
    def get_separators(image, mode='horizontal', threshold=0.1):
        """ This function looks for separators in an image `image`. By default it looks for white runs in the image by
        adding up the pixel values across the x or y dimension depending on the `mode` parameter and check if it exceeds
         a given threshold given by the parameter `threshold`. If you're looking for black runs, just invert the image.

        :param image: input image
        :param mode: can be one of 'horizontal' (0) or 'vertical' (1)
        :param threshold: the value the sum of the pixels must exceed to be defined as a white run
        :return: A list of tuples containing the row/column where a white run is present together with its relative score value.
        """
        if type(mode) == str:
            if mode.lower() == 'horizontal':
                mode = 0
            elif mode.lower() == 'vertical':
                mode = 1
        if mode not in [0, 1]:
            raise ValueError("Provide a proper mode, possible options are 'horizontal' (0) or 'vertical' (1).")

        image_height, image_width = image.shape[:2]

        separators = None
        if mode == 0:
            profiles = np.sum(image, axis=1) / 255
            separators = [(i, hp / image_width) for i, hp in enumerate(profiles) if hp / image_width > threshold]
        elif mode == 1:
            profiles = np.sum(image, axis=0) / 255
            separators = [(i, vp / image_height) for i, vp in enumerate(profiles) if vp / image_height > threshold]

        return separators

    # def get_separators(self, threshold_horizontal=0.1, threshold_vertical=0.5):
    #
    #     height, width = self.images['text_block'].shape
    #
    #     horizontal_profiles = np.sum(self.images['text_block'], axis=1) / 255
    #     vertical_profiles = np.sum(self.images['text_block'], axis=0) / 255
    #
    #     # We check for '<', because we search for blackruns in the textblock netoutput!
    #     horizontal_separators = [(i, hp / width) for i, hp in enumerate(horizontal_profiles) if
    #                              hp / width < threshold_horizontal]
    #     vertical_separators = [(i, vp / height) for i, vp in enumerate(vertical_profiles) if
    #                            vp / height < threshold_vertical]
    #
    #     print(len(horizontal_separators))
    #     print(horizontal_separators)
    #     print(len(vertical_separators))
    #     print(vertical_separators)
    #
    #     return horizontal_separators, vertical_separators

    def run_recursion(self, region_rectangle: Rectangle, max_recursion_depth=MAX_RECURSION_DEPTH, mode="horizontal", threshold=0.9):
        """ Run recursion to determine the text regions. Make sure to alternate between horizontal and vertical
        separator detection. The `mode` parameter determines with which subdivision to start, defaults to 'horizontal'.

        :param region_rectangle: determines the region in the original text block image
        :param threshold: relative number of white pixels that should be reached to be defined as a white run.
        :param mode: same parameter as in method `get_separators`, 'horizontal' or 'vertical'.
        :param max_recursion_depth: maximal number of times to run the recursion
        :return: a mask that can be applied to the baseline detection output to get a division into text regions
        """
        print(MAX_RECURSION_DEPTH - max_recursion_depth)

        if max_recursion_depth == 0:
            return

        image = self.images["text_block"]

        image = image[region_rectangle.x: region_rectangle.x + region_rectangle.width][
                region_rectangle.y: region_rectangle.y + region_rectangle.height]

        # The min_pixel_separator_distance determines up to which (pixel)distance neighboring white runs get merged!
        min_pixel_separator_distance = int(self.image_height * MIN_PIXEL_SEPARATOR_DISTANCE_FACTOR)
        print(f"min_pixel_separator_distance = {min_pixel_separator_distance}")

        # profile_list = self.get_separators(255 - self.images['text_block'], mode, threshold)
        profile_list = self.get_separators(255 - image, mode, threshold)
        index_separators = [i for i, _ in profile_list]
        if not index_separators:
            return

        index_separators_new = []
        if index_separators[0] > min_pixel_separator_distance:
            index_separators_new.append((0, index_separators[0]))

        for i in range(len(index_separators) - 1):
            if index_separators[i + 1] - index_separators[i] > min_pixel_separator_distance:
                index_separators_new.append((index_separators[i] + 1, index_separators[i + 1]))
        if mode == 'horizontal':
            if (self.image_height - 1) - index_separators[-1] > min_pixel_separator_distance:
                index_separators_new.append((index_separators[-1], self.image_height - 1))
        elif mode == 'vertical':
            if (self.image_width - 1) - index_separators[-1] > min_pixel_separator_distance:
                index_separators_new.append((index_separators[-1], self.image_width - 1))
        # print(index_separators)
        # print(index_separators_new)

        new_mode = None
        if mode == "horizontal":
            new_mode = "vertical"
        elif mode == "vertical":
            new_mode = "horizontal"

        new_region_rectangle = None
        for image_range in index_separators_new:
            # image_range is a tuple with x coordinate from
            # new_region_rectangle = None
            if mode == "horizontal":
                # update the y-coordinates and keep the x-coordinates
                new_y = image_range[0] + region_rectangle.y
                new_height = image_range[1] - image_range[0]
                new_region_rectangle = Rectangle(region_rectangle.x, new_y, region_rectangle.width, new_height)
            elif mode == "vertical":
                # update the x-coordinates and keep the y-coordinates
                new_x = image_range[0] + region_rectangle.x
                new_width = image_range[1] - image_range[0]
                new_region_rectangle = Rectangle(new_x, region_rectangle.y, new_width, region_rectangle.height)
            print("REGION RECTANGLE COORD: ", new_region_rectangle.get_vertices())
            cv2.rectangle(self.images["empty_image"], new_region_rectangle.get_vertices()[0], new_region_rectangle.get_vertices()[2], (255, 0, 0), 1)
            # self.get_separators(self.images["text_block"][image_range[0]:image_range[1]], new_mode, threshold)
            self.run_recursion(new_region_rectangle, max_recursion_depth - 1, new_mode, max(0.9*threshold, 0.65))

        return new_region_rectangle

    def run(self):
        rotation_angle = round(self.get_best_rotation_angle(), 4)
        self.rotate_images(rotation_angle)

        region_rectangle_image = Rectangle(0, 0, self.image_width, self.image_height)
        self.run_recursion(region_rectangle_image, threshold=0.9)

        plt.set_cmap('gray')
        plt.subplot(1, 3, 1)
        plt.imshow(self.images["empty_image"])
        plt.subplot(1, 3, 2)
        plt.imshow(self.images["text_block"])
        plt.subplot(1, 3, 3)
        plt.imshow(self.images["original_image"])

        plt.show()



if __name__ == '__main__':
    path_to_image_folder = '/home/max/devel/projects/python/article_separation/data/test_post_processing/textblock/'
    path_to_orig_image = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004.jpg')
    path_to_tb_outline = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004_OUT0.jpg')
    path_to_tb = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004_OUT1.jpg')
    path_to_separator = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004_OUT2.jpg')

    orig_image = cv2.imread(path_to_orig_image, cv2.IMREAD_UNCHANGED)
    tb_outline_image = cv2.imread(path_to_tb_outline, cv2.IMREAD_UNCHANGED)
    tb_image = cv2.imread(path_to_tb, cv2.IMREAD_UNCHANGED)
    separator_image = cv2.imread(path_to_separator, cv2.IMREAD_UNCHANGED)

    orig_image = cv2.resize(orig_image, None, fx=0.4, fy=0.4)
    # orig_image_gb = cv2.GaussianBlur(orig_image, (5, 5), 0)
    orig_image_gb = orig_image
    _, orig_image_gb_bin = cv2.threshold(orig_image_gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tb_pp = TextBlockNetPostProcessor(orig_image_gb_bin, tb_outline_image, tb_image, separator_image)

    region_rectangle_image = Rectangle(0, 0, orig_image.shape[1], orig_image.shape[0])
    # tb_pp.run_recursion(region_rectangle_image)
    #
    # text_block_rgb = cv2.cvtColor(tb_pp.images["text_block"], cv2.COLOR_BGR2RGB)
    # # text_block_rgb = tb_pp.images["text_block"]
    # plt.imshow(text_block_rgb)
    # plt.show()

    tb_pp.run()

    # # CONTOURS TEST
    # original_image_rgb = cv2.cvtColor(tb_pp.images["original_image"], cv2.COLOR_BGR2RGB)
    # text_block_image_rgb = cv2.cvtColor(tb_pp.images["text_block"], cv2.COLOR_BGR2RGB)
    # plt.subplot(1, 2, 1)
    # plt.imshow(text_block_image_rgb)
    # contours, _ = cv2.findContours(tb_pp.images["text_block"], cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # contour_image = cv2.drawContours(text_block_image_rgb, contours, -1, (0, 255, 0), 3)
    # plt.subplot(1, 2, 2)
    # plt.imshow(text_block_image_rgb)
    # plt.show()

    # rotation_angle = round(tb_pp.get_best_rotation_angle(), 4)
    # tb_pp.rotate_images(rotation_angle)
    #
    # horizontal_profile_list, vertical_profile_list = tb_pp.get_separators()
    #
    # index_horizontal = [i for i, _ in horizontal_profile_list]
    # index_vertical = [i for i, _ in vertical_profile_list]
    #
    # white_sep = np.zeros(orig_image.shape, dtype=np.uint8)
    # white_sep[:, index_vertical] = 255
    # white_sep[index_horizontal, :] = 255
    # # white_sep = cv2.resize(white_sep, None, fx=0.5, fy=0.5)
    #
    # # separator_image = cv2.resize(separator_image, None, fx=0.5, fy=0.5)
    # separator_image = np.array((separator_image > 0.2), np.uint8)
    # print(separator_image, separator_image.dtype)
    # separator_image *= 255
    #
    # print(separator_image, separator_image.dtype)
    # print(white_sep, white_sep.dtype)
    #
    # add_condition = np.not_equal(white_sep, separator_image)
    # black_white_separator = np.copy(white_sep)
    # black_white_separator[add_condition] += separator_image[add_condition]
    #
    # kernel = np.ones((5, 5), np.uint8)
    # black_white_separator = cv2.morphologyEx(black_white_separator, cv2.MORPH_CLOSE, kernel)
    #
    # plt.set_cmap("gray")
    # plt.subplot(1, 4, 1)
    # plt.imshow(white_sep)
    # plt.subplot(1, 4, 2)
    # plt.imshow(separator_image)
    # plt.subplot(1, 4, 3)
    # plt.imshow(black_white_separator)
    # plt.subplot(1, 4, 4)
    # plt.imshow(orig_image)
    # plt.show()
    #
    # # cv2.imshow('white separator', white_sep)
    # # cv2.imshow('black separator net', separator_image)
    # # cv2.imshow('black white separator', black_white_separator)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    #
    # vertical_profile = np.sum(black_white_separator, axis=0)
    # horizontal_profile = np.sum(black_white_separator, axis=1)
    #
    # horizontal = [(i, hp / orig_image.shape[1] / 255) for i, hp in enumerate(horizontal_profile) if
    #               hp / orig_image.shape[1] / 255 < 0.2]
    # vertical = [(i, vp / orig_image.shape[0] / 255) for i, vp in enumerate(vertical_profile) if
    #             vp / orig_image.shape[0] / 255 < 0.2]
    #
    # horizontal_index = [i for i, _ in horizontal]
    # vertical_index = [i for i, _ in vertical]
    #
    # print(horizontal_index)
    # print(vertical_index)
    #
    #
    # def convert_to_ranges(index_list):
    #     range_list = []
    #     skip = False
    #     for i in range(len(index_list) - 1):
    #         if not skip:
    #             begin = index_list[i]
    #         if index_list[i + 1] - index_list[i] < 3:
    #             skip = True
    #             continue
    #         skip = False
    #         end = index_list[i]
    #         range_list.append((begin, end))
    #     return range_list
    #
    #
    # print(convert_to_ranges(horizontal_index))
    # print(convert_to_ranges(vertical_index))
    #
    # # tb_image_binarized = np.array((tb_image > 0.8), np.uint8) * 255
    # # print(tb_image_binarized)
    # # # erosion_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    # # erosion_kernel = np.ones([8, 8], dtype=np.uint8)
    # # print(erosion_kernel)
    # # tb_image_erosion = cv2.erode(tb_image_binarized, erosion_kernel, iterations=1)
    # # tb_image_erosion = cv2.resize(tb_image_erosion, None, fx=0.4, fy=0.4)
    # # print(tb_image_erosion)
    # # cv2.imshow("erosion image textblock", tb_image_erosion)
    # # cv2.waitKey(0)
    # # cv2.destroyAllWindows()
    # # exit(1)
