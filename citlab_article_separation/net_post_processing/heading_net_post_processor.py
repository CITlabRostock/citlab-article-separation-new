import argparse
from collections import Counter

import numpy as np

from citlab_article_separation.net_post_processing.net_post_processing_helper import load_and_scale_image, \
    get_net_output
from citlab_article_separation.net_post_processing.region_net_post_processor_base import RegionNetPostProcessor
from citlab_article_separation.net_post_processing.region_to_page_writer import RegionToPageWriter
from citlab_python_util.image_processing.swt_dist_trafo import StrokeWidthDistanceTransform
from citlab_python_util.io.file_loader import get_page_path
from citlab_python_util.logging.custom_logging import setup_custom_logger
from citlab_python_util.parser.xml.page.page_constants import TextRegionTypes
from citlab_python_util.parser.xml.page.page_objects import TextLine

logger = setup_custom_logger("HeadingNetPostProcessor", "info")


class HeadingNetPostProcessor(RegionNetPostProcessor):
    def __init__(self, image_list, path_to_pb, fixed_height, scaling_factor, weight_dict=None, threshold=0.5,
                 thresh_dict=None, text_line_percentage=None):
        """
        :param image_list:
        :param path_to_pb:
        :param fixed_height:
        :param scaling_factor:
        :param weight_dict: contains the weights for the different header features, supports "net", "stroke_width" and "text_height"
        """
        super().__init__(image_list, path_to_pb, fixed_height, scaling_factor)
        self.SWT = StrokeWidthDistanceTransform(dark_on_bright=True)
        self.weight_dict = weight_dict if weight_dict is not None else {"net": 0.33, "stroke_width": 0.33,
                                                                        "text_height": 0.33}
        self.threshold = threshold
        self.thresh_dict = thresh_dict if thresh_dict is not None else {"net_thresh": 0.9, "stroke_width_thresh": 0.9,
                                                                        "text_height_thresh": 0.9, "sw_th_thresh": 0.8}
        self.text_line_percentage = text_line_percentage if text_line_percentage is not None else 1.0

    def scale_to_new_interval(self, data, old_min, old_max, new_min=0, new_max=1):
        if old_max - old_min == 0:
            return data
        return (new_max - new_min) / (old_max - old_min) * (data - old_min) + new_min

    def to_page_xml(self, page_path, image_path=None, net_output_post=None, swt_feature_image=None, *args, **kwargs):
        region_page_writer = RegionToPageWriter(page_path, path_to_image=image_path, fixed_height=self.fixed_height,
                                                scaling_factor=self.scaling_factor)
        # image_width, image_height = get_image_dimensions(image_path)

        swt_feature_image = self.get_swt_features_image(image_path)
        text_lines = region_page_writer.page_object.get_textlines()

        # Get features for each text line
        text_line_stroke_width_dict = dict()
        text_line_height_dict = dict()
        text_line_net_prob_dict = dict()
        for text_line in text_lines:
            if text_line.surr_p is None:
                text_line_stroke_width = 0
                text_line_height = 0
            else:
                text_line_stroke_width, text_line_height = self.get_swt_features_textline(swt_feature_image, text_line)

            text_line_stroke_width_dict[text_line.id] = text_line_stroke_width
            text_line_height_dict[text_line.id] = text_line_height
            if self.weight_dict['net'] == 0 or net_output_post is None:
                text_line_net_prob_dict[text_line.id] = 0
            else:
                text_line_net_prob_dict[text_line.id] = self.get_net_prob_for_text_line(net_output_post, text_line,
                                                                                        region_page_writer.scaling_factor)

        # Get the most common stroke width and text height (median vs mode - or combination of both?)
        text_line_stroke_width_list = list(text_line_stroke_width_dict.values())


        if len(text_line_stroke_width_list) == 0:
            use_swt_features = False
        else:
            use_swt_features = True

            stroke_width_mode = Counter(text_line_stroke_width_list).most_common(1)[0][0]
            # stroke_width_median = np.median(text_line_stroke_width_list)
            text_line_height_list = list(text_line_height_dict.values())
            text_line_height_mode = Counter(text_line_height_list).most_common(1)[0][0]
            # text_line_height_median = np.median(text_line_height_list)

            for text_line in text_lines:
                # text_line_stroke_width_diff = text_line_stroke_width_dict[text_line.id] - stroke_width_median
                text_line_stroke_width_diff = text_line_stroke_width_dict[text_line.id] - stroke_width_mode
                # text_line_stroke_width_diff /= image_width
                # text_line_stroke_width_diff_scaled = text_line_stroke_width_diff
                text_line_stroke_width_dict[text_line.id] = text_line_stroke_width_diff

                # text_line_height_diff = text_line_height_dict[text_line.id] - text_line_height_median
                text_line_height_diff = text_line_height_dict[text_line.id] - text_line_height_mode
                # text_line_height_diff /= image_height
                # text_line_height_diff_scaled = text_line_height_diff
                text_line_height_dict[text_line.id] = text_line_height_diff

            text_line_stroke_width_list = list(text_line_stroke_width_dict.values())
            stroke_width_min = np.min(text_line_stroke_width_list)
            stroke_width_max = np.max(text_line_stroke_width_list)

            text_line_height_list = list(text_line_height_dict.values())
            text_line_height_min = np.min(text_line_height_list)
            text_line_height_max = np.max(text_line_height_list)

            net_weight = self.weight_dict["net"]
            stroke_width_weight = self.weight_dict["stroke_width"]
            text_line_height_weight = self.weight_dict["text_height"]

            net_thresh = self.thresh_dict["net_thresh"]
            stroke_width_thresh = self.thresh_dict["stroke_width_thresh"]
            text_line_height_thresh = self.thresh_dict["text_height_thresh"]
            sw_th_thresh = self.thresh_dict["sw_th_thresh"]

        page_object = region_page_writer.page_object
        for text_line in text_lines:
            text_line_net_conf = text_line_net_prob_dict[text_line.id]
            if use_swt_features:
                text_line_stroke_width_conf = self.scale_to_new_interval(text_line_stroke_width_dict[text_line.id],
                                                                         old_min=stroke_width_min,
                                                                         old_max=stroke_width_max)
                text_line_height_conf = self.scale_to_new_interval(text_line_height_dict[text_line.id],
                                                                   old_min=text_line_height_min,
                                                                   old_max=text_line_height_max)

                if text_line_stroke_width_conf >= stroke_width_thresh or text_line_height_conf >= text_line_height_thresh or \
                        (text_line_stroke_width_conf + text_line_height_conf) / 2 >= sw_th_thresh or text_line_net_conf >= net_thresh:
                    is_heading_confidence = 1.0

                else:
                    is_heading_confidence = net_weight * text_line_net_conf \
                                            + stroke_width_weight * text_line_stroke_width_conf \
                                            + text_line_height_weight * text_line_height_conf
            else:
                is_heading_confidence = text_line_net_conf

            if is_heading_confidence > self.threshold:
                text_line_nd = page_object.get_child_by_id(page_object.page_doc, text_line.id)[0]
                page_object.set_custom_attr(text_line_nd, "structure", "semantic_type", TextRegionTypes.sHEADING)

        for text_region in region_page_writer.page_object.get_text_regions():
            page_nd = page_object.get_child_by_id(page_object.page_doc, text_region.id)[0]
            page_nd.set("type", TextRegionTypes.sPARAGRAPH)

            if not text_region.text_lines:
                continue

            num_text_line_headings = 0
            for text_line in text_region.text_lines:
                if 'structure' in text_line.custom and 'semantic_type' in text_line.custom['structure']:
                    if text_line.custom['structure']['semantic_type'] == TextRegionTypes.sHEADING:
                        num_text_line_headings += 1

            # if num_text_line_headings > 0:
            # If one text line is a heading the whole region gets classified as a heading
            if num_text_line_headings / len(text_region.text_lines) >= self.text_line_percentage:
                page_nd.set('type', TextRegionTypes.sHEADING)

        logger.debug(f"Saving HeadingNetPostProcessor results to page {page_path}")
        region_page_writer.save_page_xml(page_path + ".xml")

        return region_page_writer.page_object

    def post_process(self, net_output):
        """
        Take as input the neural net output and the SWT-features and combine them to get the header regions.
        :param net_output:
        :return:
        """
        # Ignore the other class
        return net_output[:, :, 0] / 255

    def get_swt_features_image(self, image_path):
        """
        Apply a stroke width transformation (SWT) to the input image and use the features to extract the header regions.
        :return:
        """
        return self.SWT.distance_transform(image_path)

    def get_swt_features_textline(self, swt_feature_image, text_line: TextLine):
        """
        For a given line object `text_line` return the corresponding stroke width of the text.

        :param swt_feature_image: SWT feature image generated by the function `get_swt_features`.
        :param text_line: TextLine object
        :return:
        """
        bounding_box = text_line.surr_p.to_polygon().get_bounding_box()
        xa, xb = bounding_box.x, bounding_box.x + bounding_box.width
        ya, yb = bounding_box.y, bounding_box.y + bounding_box.height

        text_line_swt = swt_feature_image[ya:yb + 1, xa:xb + 1]

        text_line_ccs = self.SWT.connected_components_cv(text_line_swt)
        text_line_ccs = self.SWT.clean_connected_components(text_line_ccs)

        swt_cc_values = []
        text_line_height = 0
        for cc in text_line_ccs:
            swt_cc_values.append(np.max(text_line_swt[cc[1]: cc[1] + cc[3], cc[0]: cc[0] + cc[2]]))
            if cc[3] > text_line_height:
                text_line_height = cc[3]

        text_line_stroke_width = np.median(swt_cc_values) if swt_cc_values else 0.0

        return text_line_stroke_width, text_line_height

    def get_net_prob_for_text_line(self, net_output, text_line, scaling_factor):
        # since we use a scaling factor for the net_output, we need a rescaled bounding box
        if text_line.surr_p is None:
            return 0
        text_line_polygon = text_line.surr_p.to_polygon()
        text_line_polygon.rescale(scaling_factor)
        bounding_box = text_line_polygon.get_bounding_box()
        xa, xb = bounding_box.x, bounding_box.x + bounding_box.width
        ya, yb = bounding_box.y, bounding_box.y + bounding_box.height

        net_output_text_line = net_output[ya:yb, xa:xb]
        prob_sum = np.sum(net_output_text_line)

        return prob_sum / (bounding_box.width * bounding_box.height)

    def run(self, gpu_device='0'):
        new_page_objects = []
        for image_path in self.image_paths:
            image, image_grey, sc = load_and_scale_image(image_path, self.fixed_height, self.scaling_factor)
            self.images.append(image)

            # net_output has shape HWC
            if self.weight_dict['net'] > 0:
                net_output = get_net_output(image_grey, self.pb_graph, gpu_device)
                net_output = np.array(net_output * 255, dtype=np.uint8)
                self.net_outputs.append(net_output)

                # remove the "garbage" channel
                net_output_post = self.post_process(net_output)
                self.net_outputs_post.append(net_output_post)
            else:
                net_output_post = None

            # get swt feature image and use it for heading classification
            swt_feature_image = self.get_swt_features_image(image_path)

            page_object = self.to_page_xml(get_page_path(image_path), image_path, net_output_post, swt_feature_image)

            new_page_objects.append(page_object)

        return new_page_objects


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str, required=True,
                        help="Path to the image list for which separator information should be created.")
    parser.add_argument('--path_to_pb', type=str, required=True,
                        help="Path to the TensorFlow pb graph for creating the separator information")
    parser.add_argument('--fixed_height', type=int, required=False,
                        help="If parameter is given, the images will be scaled to this height by keeping the aspect "
                             "ratio", default=None)
    parser.add_argument('--scaling_factor', type=float, required=False,
                        help="If no --fixed_height flag is given, use a predefined scaling factor on the images.",
                        default=1.0)
    parser.add_argument('--threshold', type=float, required=False, default=0.4,
                        help="Threshold value that decides based on the feature values if a text line is a heading or "
                             "not.")
    parser.add_argument('--net_weight', type=float, required=False, help="Weight the net output feature.")
    parser.add_argument('--stroke_width_weight', type=float, required=False, help="Weight the stroke width feature.")
    parser.add_argument('--text_height_weight', type=float, required=False, help="Weight the text line height feature.")
    parser.add_argument('--gpu_devices', type=str, required=False,
                        help='Which GPU devices to use, comma-separated integers. E.g. "0,1,2".',
                        default='0')
    parser.add_argument("--net_thresh", type=float, required=False, help="If the net confidence is greater than or "
                                                                         "equal to this value the text line is "
                                                                         "considered a heading.")
    parser.add_argument("--stroke_width_thresh", type=float, required=False,
                        help="If the stroke width confidence is greater than or equal to his value the text line is"
                             "considered a heading.")
    parser.add_argument("--text_height_thresh", type=float, required=False,
                        help="If the text height confidence is greater than or equal to this value the text line is"
                             "considered a heading.")
    parser.add_argument("--sw_th_thresh", type=float, required=False,
                        help="If the average of stroke width and text height confidence is greater than or equal to "
                             "this value the text line is considered a heading.")
    parser.add_argument("--text_line_percentage", type=float, required=False,
                        help="Declare a region as heading if text_line_percentage percent text lines are considered as \
                        headings.")

    args = parser.parse_args()

    gpu_devices = args.gpu_devices
    image_list = args.image_list
    path_to_pb = args.path_to_pb
    fixed_height = args.fixed_height
    scaling_factor = args.scaling_factor

    if args.net_weight is None or args.stroke_width_weight is None or args.text_height_weight is None:
        weight_dict = None
    else:
        weight_dict = {"net": args.net_weight,
                       "stroke_width": args.stroke_width_weight,
                       "text_height": args.text_height_weight}

    if args.net_thresh is None or args.stroke_width_thresh is None or args.text_height_thresh is None or args.sw_th_thresh is None:
        thresh_dict = None
    else:
        thresh_dict = {"net_thresh": args.net_thresh,
                       "stroke_width_thresh": args.stroke_width_thresh,
                       "text_height_thresh": args.text_height_thresh,
                       "sw_th_thresh": args.sw_th_thresh}
    threshold = args.threshold
    text_line_percentage = args.text_line_percentage

    post_processor = HeadingNetPostProcessor(image_list, path_to_pb, fixed_height, scaling_factor, weight_dict,
                                             threshold, thresh_dict, text_line_percentage)
    post_processor.run(gpu_devices)
