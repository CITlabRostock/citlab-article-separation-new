from abc import ABC, abstractmethod

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from PIL import Image
from matplotlib.collections import PolyCollection

from citlab_article_separation.net_post_processing.net_post_processing_helper import load_image_paths, \
    load_and_scale_image, load_graph, get_net_output, apply_threshold
from citlab_python_util.geometry.point import rescale_points
from citlab_python_util.geometry.util import alpha_shape
from citlab_python_util.io.file_loader import get_page_path
from citlab_python_util.parser.xml.page.plot import plot_pagexml


class RegionNetPostProcessor(ABC):
    def __init__(self, image_list, path_to_pb, fixed_height, scaling_factor, threshold=None, gpu_devices='0'):
        if type(image_list) == str:
            self.image_paths = load_image_paths(image_list)
        else:
            self.image_paths = image_list
        self.fixed_height = fixed_height
        self.scaling_factor = scaling_factor
        self.threshold = threshold

        self.pb_graph = load_graph(path_to_pb)

        self.images = []
        self.net_outputs = []
        self.net_outputs_post = []

        self.gpu_devices = gpu_devices

        # self.net_output_polygons = []

    def run(self):
        for image_path in self.image_paths:
            image, image_grey, sc = load_and_scale_image(image_path, self.fixed_height, self.scaling_factor)
            self.images.append(image)

            # net_output has shape HWC
            net_output = get_net_output(image_grey, self.pb_graph, gpu_device=self.gpu_devices)
            net_output = np.array(net_output * 255, dtype=np.uint8)
            self.net_outputs.append(net_output)
            net_output = apply_threshold(net_output, self.threshold)

            # The post processing depends on the region to be saved (TextRegion, SeparatorRegion, ImageRegion, ...)
            net_output_post = self.post_process(net_output)

            self.net_outputs_post.append(net_output_post)

            # plot_pagexml(get_page_path(image_path), image_path, plot_article=False, plot_legend=False,
            #              fill_regions=True,
            #              use_page_image_resolution=True)
            # plt.show()

            # since there can be multiple region types put them in a dictionary
            polygons_dict = self.to_polygons(net_output_post)

            # upscale polygons to the original image size
            polygons_dict = self.rescale_polygons(polygons_dict, scaling_factor=1/sc)

            # self.plot_polygons(image, polygons_dict["SeparatorRegion"])

            page_object = self.to_page_xml(get_page_path(image_path), image_path=image_path, polygons_dict=polygons_dict)
            # plot_pagexml(page_object, image_path, plot_article=False, plot_legend=False, fill_regions=True,
            #              use_page_image_resolution=True)
            # plt.show()

            # self.net_output_polygons.append(polygons)
            # self.plot_polygons(image, self.net_output_polygons[-1])

            # self.plot_binary(image, net_output_post)

    def to_polygons(self, binary_image):
        """
        Converts a binary image (the net output) to polygons that can later be saved to the PAGE files.
        :return: dictionary of list of polygons, e.g. "{TextRegion: list of polygons}"
        """
        pass

    def plot_polygons(self, image, polygons):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.imshow(image)

        poly_collection = PolyCollection(polygons, closed=True, edgecolors='red', facecolors='yellow', linewidths=2,
                                         alpha=0.5)
        ax.add_collection(poly_collection)
        plt.show()

    def plot_binary(self, image, net_output):
        image_pil = Image.fromarray(image).convert('RGBA')
        cm = plt.get_cmap('Reds')
        net_output_threshold = Image.fromarray(apply_threshold(net_output, 0.4)).convert('L')
        net_output = np.uint8(cm(net_output) * 255)
        net_output_pil = Image.fromarray(net_output).convert('RGBA')
        # net_output_pil.putalpha(128)

        # blended_image = Image.blend(image_pil, net_output_pil, alpha=0.5)
        # alpha_composite = Image.alpha_composite(image_pil, net_output_pil)
        blended_image = image_pil.copy()
        blended_image.paste(net_output_pil, mask=net_output_threshold)
        blended_image = np.array(blended_image, np.uint8)
        plt.imshow(blended_image)
        # plt.imshow(np.array(image_pil, np.uint8))
        plt.show()
        # blended_image.show()

    @abstractmethod
    def to_page_xml(self, page_path, image_path=None, *args, **kwargs):
        pass

    def remove_every_nth_point(self, polygon, n=2, min_num_points=20, iterations=1):
        if iterations <= 0:
            return polygon
        if len(polygon) // n < min_num_points:
            return polygon
        res = polygon[::n]
        if polygon[0] == polygon[-1] and res[0] != res[-1]:
            res.append(res[0])

        return self.remove_every_nth_point(res, n, min_num_points, iterations - 1)

    def apply_contour_detection(self, image, use_alpha_shape=False):
        binary_image = image

        # _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
        # contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = rasterio.features.shapes(binary_image, connectivity=8)
        contours = [p[0]['coordinates'][0] for p in contours if p[1] == 255]
        if use_alpha_shape:
            contours = [alpha_shape(np.array(polygon), alpha=0.1) for polygon in contours]

        return contours

    def apply_contour_detection2(self, binary_image):
        """
        Given a binary image `binary_image` the contours are calculated. This can result in Polygons with outer AND
        inner points
        :param binary_image:
        :return:
        """
        contours = rasterio.features.shapes(binary_image, connectivity=8)
        contours = [p[0]['coordinates'] for p in contours if p[1] == 255]

        return contours

    @abstractmethod
    def post_process(self, net_output):
        pass

    def apply_morphology_operators(self, image, kernel=None):
        if kernel is None:
            kernel = np.ones([8, 1], np.uint8)

        # kernel_opening = np.ones([5, 10], np.uint8)
        # kernel_closing = np.ones([10, 2], np.uint8)

        kernel_opening = kernel_closing = kernel

        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_opening)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_closing)

        return image

    def apply_cc_analysis(self, net_output, threshold):
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(net_output, connectivity=8)
        sizes = stats[1:, -1]
        nb_components = nb_components - 1

        min_size = int(net_output.size * threshold)

        net_output_new = np.zeros(output.shape, net_output.dtype)
        for i in range(0, nb_components):
            if sizes[i] >= min_size:
                net_output_new[output == i + 1] = 255

        return net_output_new

    def rescale_polygons(self, polygons_dict, scaling_factor):
        for region_name, polygon_list in polygons_dict.items():
            new_polygon_list = []
            for polygon in polygon_list:
                new_polygon_list.append([rescale_points(poly, scaling_factor) for poly in polygon])
            # polygons_dict[region_name] = [rescale_points(polygon, scaling_factor) for polygon in polygon_list]
            polygons_dict[region_name] = new_polygon_list

        return polygons_dict


if __name__ == '__main__':
    # ONB Test Set (3000 height)
    image_list = "/home/max/data/la/textblock_detection/newseye_tb_data/onb/tmp.lst"
    # Independance_lux dataset (3000 height)
    # image_list = '/home/max/data/la/textblock_detection/bnl_data/independance_lux/traindata_headers/val.lst'

    # Textblock detection
    path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
                 "racetrack_onb_textblock_136/TB_aru_3000_height_scaling_train_only/export/" \
                 "TB_aru_3000_height_scaling_train_only_2020-06-05.pb"

    # Header detection
    # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/independance_lux/headers/" \
    #              "tb_headers_aru/export/tb_headers_aru_2020-06-04.pb"
    # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #              "racetrack_onb_textblock_136/with_headings/TB_aru_3000_height/export/TB_aru_3000_height_2020-06-10.pb"

    # Separators
    path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/separator_detection/SEP_aru_5300/export/" \
                 "SEP_aru_5300_2020-06-10.pb"
    tb_pp = RegionNetPostProcessor(image_list, path_to_pb, fixed_height=None, scaling_factor=0.4, threshold=0.05)
    # tb_pp = TextBlockNetPostProcessor2(path_to_pb, fixed_height=None, scaling_factor=0.7, threshold=0.1)
    tb_pp.run()
