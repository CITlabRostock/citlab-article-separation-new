import argparse

import numpy as np
import cv2
from citlab_python_util.io.file_loader import get_page_path
from scipy.ndimage.interpolation import rotate
from PIL import Image

from citlab_article_separation.net_post_processing.net_post_processing_helper import load_and_scale_image, \
    get_net_output, apply_threshold

from citlab_article_separation.net_post_processing.region_net_post_processor_base import RegionNetPostProcessor
from citlab_article_separation.net_post_processing.separator_region_to_page_writer import SeparatorRegionToPageWriter
from citlab_python_util.logging import custom_logging
from citlab_python_util.parser.xml.page.page_constants import sSEPARATORREGION

logger = custom_logging.setup_custom_logger("SeparatorNetPostProcessor", level="info")


class SeparatorNetPostProcessor(RegionNetPostProcessor):

    def __init__(self, image_list, path_to_pb, fixed_height, scaling_factor, threshold, gpu_devices):
        super().__init__(image_list, path_to_pb, fixed_height, scaling_factor, threshold, gpu_devices)

    def post_process(self, net_output):
        """
        Post process the raw net output `net_output`, e.g. by removing CCs that have a size smaller than 100 pixels.
        :param net_output: numpy array with dimension HWC (only channel 1 - the separator channel - is important)
        :return: post processed net output of the same dimension as the input (HWC)
        """
        # Ignore the other class
        net_output = net_output[:, :, 0]
        net_output_height, net_output_width = net_output.shape

        # Delete connected components that have a size of less than 100 pixels
        net_output_post = self.apply_cc_analysis(net_output, 1 / net_output.size * 100)

        # # Uncomment if you want to rotate the image first - attention: this slows down the process by ~1s per image
        # max_h = 0
        # max_v = 0
        # rotation_angle_v = 0
        # rotation_angle_h = 0
        # # Check for small rotations from -2 to 2 degrees in steps of 0.2
        # for angle in range(-2, 3, 1):
        #     # angle *= 0.1
        #     # rotate image, here we don't care about dimensions yet
        #     rotated_image = rotate(net_output_post, angle=angle)
        #     horizontal_projection = np.sum(rotated_image, axis=1)
        #     vertical_projection = np.sum(rotated_image, axis=0)
        #
        #     if max_h < np.max(horizontal_projection):
        #         max_h = np.max(horizontal_projection)
        #         rotation_angle_h = angle
        #     if max_v < np.max(vertical_projection):
        #         max_v = np.max(vertical_projection)
        #         rotation_angle_v = angle
        #
        # # if the rotation angle of the vertical and horizontal projection differ too much don't rotate
        # rotation_angle = 0
        # if abs(rotation_angle_v - rotation_angle_h) <= 1:
        #     rotation_angle = (rotation_angle_v + rotation_angle_h)/2
        #
        # # rotate image
        # if int(rotation_angle) != 0:
        #     net_output_post_pil: Image.Image = Image.fromarray(net_output_post)
        #     net_output_post_pil.rotate(rotation_angle, expand=True)
        #     net_output_post = np.array(net_output_post_pil, dtype=np.uint8)

        # Extract all horizontal separators
        horizontal_min_width = int(15 * net_output_width / 1000)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_min_width, 1))
        horizontal_mask = cv2.morphologyEx(net_output_post, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)

        # Extract all vertical separators
        vertical_min_height = int(30 * net_output_height / 1500)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_min_height))
        vertical_mask = cv2.morphologyEx(net_output_post, cv2.MORPH_OPEN, vertical_kernel, iterations=1)

        # Subtract the vertical_mask from the horizontal_mask in a final step
        # With this step we make sure that there are no overlapping separators in the PAGE file in the end.
        horizontal_mask = cv2.subtract(horizontal_mask, vertical_mask)

        # Remove created noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (int(10 * net_output_width / 1000), 1))
        horizontal_mask = cv2.morphologyEx(horizontal_mask, cv2.MORPH_OPEN, kernel)

        # # Rotate the images back - uncomment if you used rotation at the top
        # horizontal_mask_pil = Image.fromarray(horizontal_mask)
        # horizontal_mask_pil.rotate(-rotation_angle, expand=True)
        # horizontal_mask = np.array(horizontal_mask_pil, dtype=np.uint8)
        #
        # vertical_mask_pil = Image.fromarray(vertical_mask)
        # vertical_mask_pil.rotate(-rotation_angle, expand=True)
        # vertical_mask = np.array(vertical_mask_pil, dtype=np.uint8)

        return {"horizontal": horizontal_mask, "vertical": vertical_mask}

    def to_polygons(self, net_output, separator_type=None):
        """
        Converts the (post-processed) net output `net_output` to a list of polygons via contour detection and removal of
        unnecessary points.
        :param net_output: numpy array with dimension HWC
        :return: list of polygons representing the contours of the CCs is appended to the `net_output_polygons`
        attribute
        """
        # for net_output in self.net_outputs_post:
        contours = self.apply_contour_detection2(net_output)
        # contours = [self.remove_every_nth_point(contour, n=2, min_num_points=20, iterations=2) for contour in
        #             contours]

        if separator_type is None:
            return {sSEPARATORREGION: contours}
        else:
            return {sSEPARATORREGION + "_" + separator_type: contours}

    def to_page_xml(self, page_path, image_path=None, polygons_dict=None, *args, **kwargs):
        """
        Write the polygon information given by `polygons_dict` coming from the `to_polygons` function to the page file
        in `page_path`.
        :param **kwargs:
        :param page_path: path to the page file the region information should be written to
        :param polygons_dict: dictionary with region types as keys and the corresponding list of polygons as values
        :return: Page object that can either be further processed or be written to
        """
        # Load the region-to-page-writer and initialize it with the given page path and its region dictionary
        region_page_writer = SeparatorRegionToPageWriter(page_path, image_path, self.fixed_height,
                                                         self.scaling_factor, polygons_dict)
        region_page_writer.remove_separator_regions_from_page()
        region_page_writer.merge_regions()
        logger.debug(f"Saving SeparatorNetPostProcessor results to page {page_path}")
        region_page_writer.save_page_xml(page_path + ".xml")

        return region_page_writer.page_object

    def run(self):
        for image_path in self.image_paths:
            image, image_grey, sc = load_and_scale_image(image_path, self.fixed_height, self.scaling_factor)
            self.images.append(image)

            # net_output has shape HWC
            net_output = get_net_output(image_grey, self.pb_graph, gpu_device=self.gpu_devices)
            net_output = np.array(net_output * 255, dtype=np.uint8)
            self.net_outputs.append(net_output)
            net_output = apply_threshold(net_output, self.threshold)

            net_output_post_dict = self.post_process(net_output)

            polygons_dict = {}
            for separator_type, net_output_post in net_output_post_dict.items():
                # keys are "SeparatorRegion", "SeparatorRegion_horizontal" or "SeparatorRegion_vertical"
                polygons_dict.update(self.to_polygons(net_output_post, separator_type))
            polygons_dict = self.rescale_polygons(polygons_dict, scaling_factor=1 / sc)
            page_object = self.to_page_xml(get_page_path(image_path), image_path=image_path,
                                           polygons_dict=polygons_dict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str, required=False,
                        help="Path to the image list for which separator information should be created.")
    parser.add_argument('--path_to_pb', type=str, required=False,
                        help="Path to the TensorFlow pb graph for creating the separator information")
    parser.add_argument('--fixed_height', type=int, required=False,
                        help="If parameter is given, the images will be scaled to this height by keeping the aspect "
                             "ratio", default=1500)
    parser.add_argument('--scaling_factor', type=float, required=False,
                        help="If no --fixed_height flag is given, use a predefined scaling factor on the images.",
                        default=1.0)
    parser.add_argument('--threshold', type=float, required=False,
                        help="Threshold value that is used to convert the probability outputs of the neural network"
                             "to 0 and 1 values", default=0.05)
    parser.add_argument('--gpu_devices', type=str, required=False,
                        help='Which GPU devices to use, comma-separated integers. E.g. "0,1,2".',
                        default='')

    args = parser.parse_args()

    image_list = args.image_list
    path_to_pb = args.path_to_pb
    fixed_height = args.fixed_height
    scaling_factor = args.scaling_factor
    threshold = args.threshold
    gpu_devices = args.gpu_devices

    post_processor = SeparatorNetPostProcessor(image_list, path_to_pb, fixed_height, scaling_factor, threshold,
                                               gpu_devices)
    post_processor.run()

    # /home/max/data/as/NewsEye_ONB_232_textblocks/images.lst


    # # ONB Test Set (3000 height)
    # # image_list = "/home/max/data/la/textblock_detection/newseye_tb_data/onb/tmp.lst"
    # # Independance_lux dataset (3000 height)
    # # image_list = '/home/max/data/la/textblock_detection/bnl_data/independance_lux/traindata_headers/val.lst'
    # image_list = "/home/max/sanomia_turusta_544852/images.lst"
    #
    # # Textblock detection
    # path_to_pb_tb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #                 "racetrack_onb_textblock_136/TB_aru_3000_height_scaling_train_only/export/" \
    #                 "TB_aru_3000_height_scaling_train_only_2020-06-05.pb"
    #
    # # Header detection
    # # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/independance_lux/headers/" \
    # #              "tb_headers_aru/export/tb_headers_aru_2020-06-04.pb"
    # path_to_pb_hd = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #                 "racetrack_onb_textblock_136/with_headings/TB_aru_3000_height/export/TB_aru_3000_height_2020-06-10.pb"
    #
    # # Separators
    # path_to_pb_sp = "/home/max/devel/projects/python/aip_pixlab/models/separator_detection/SEP_aru_5300/export/" \
    #                 "SEP_aru_5300_2020-06-10.pb"
    #
    # # Comparison dbscan vs pixellabeling
    # # image_list = "/home/max/sanomia_turusta_544852/images.lst"
    # # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_tb, fixed_height=1650, scaling_factor=1.0, threshold=0.2)
    #
    # image_list = "/home/max/separator_splits_with_words/images.lst"
    #
    # # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_tb, fixed_height=None, scaling_factor=0.55, threshold=0.2)
    # # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_hd, fixed_height=None, scaling_factor=0.55, threshold=0.2)
    # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_sp, fixed_height=None, scaling_factor=1.0, threshold=0.05)
    # tb_pp.run()
