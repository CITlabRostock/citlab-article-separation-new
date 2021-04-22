import os
import shutil
from abc import ABC, abstractmethod

from citlab_python_util.io import file_loader
from citlab_python_util.image_processing.image_stats import get_image_dimensions, get_scaling_factor
from citlab_python_util.logging import custom_logging
from citlab_python_util.parser.xml.page.page import Page

logger = custom_logging.setup_custom_logger(__name__)


class RegionToPageWriter(ABC):
    def __init__(self, path_to_page, path_to_image=None, fixed_height=None, scaling_factor=None, *args, **kwargs):
        self.scaling_factor = None
        if path_to_image is not None:
            image_width, image_height = get_image_dimensions(path_to_image)
            self.scaling_factor = get_scaling_factor(image_height, image_width, scaling_factor, fixed_height)
        self.path_to_page = path_to_page
        self.page_object = self.load_page_object(path_to_page, path_to_image)

    def load_page_object(self, path_to_page, path_to_image):
        # if PAGE file not existent, create one
        if not os.path.exists(path_to_page):
            image_width, image_height = get_image_dimensions(path_to_image)
            return Page(img_filename=path_to_image, img_w=int(self.scaling_factor * image_width),
                        img_h=int(self.scaling_factor * image_height))
        return Page(path_to_page)

    def save_page_xml(self, save_path):
        self.page_object.write_page_xml(save_path)

    # def write_regions(self, create_backup=False):
    #     """
    #     Update and write the new regions into the PAGE file. IF `create_backup` is True, a backup of the original PAGE
    #     file is created by appending the ".bak" extension.
    #     """
    #     for i in range(len(self.page_object_list)):
    #         page_obj = self.page_object_list[i]
    #         new_region_dict = self.new_region_list[i]
    #         save_path = self.page_path_list[i]
    #         if create_backup:
    #             shutil.copyfile(self.page_path_list[i], self.page_path_list[i] + ".bak")
    #         for region_type, region_list in new_region_dict.items():
    #             page_obj.remove_regions(region_type)
    #             for region in region_list:
    #                 page_obj.add_region(region)
    #         page_obj.write_page_xml(save_path)
