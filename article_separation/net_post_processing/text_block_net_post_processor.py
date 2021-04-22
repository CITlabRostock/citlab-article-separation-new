from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt

from citlab_article_separation.net_post_processing.region_net_post_processor_base import RegionNetPostProcessor


class TextBlockNetPostProcessor(RegionNetPostProcessor):
    def __init__(self, image_list, path_to_pb, fixed_height, scaling_factor, threshold):
        super().__init__(image_list, path_to_pb, fixed_height, scaling_factor, threshold)

    def post_process(self, net_output):
        # net_output = self.apply_morphology_operators(net_output)
        net_output = net_output[:, :, 0]
        net_output_post = self.apply_cc_analysis(net_output, 1/net_output.size * 100)

        return net_output_post

    def to_polygons(self, net_output):
        contours = self.apply_contour_detection(net_output, use_alpha_shape=False)
        contours = [self.remove_every_nth_point(contour, n=2, min_num_points=20, iterations=1) for contour in
                    contours]
        self.net_output_polygons.append(contours)

if __name__ == '__main__':
    # ONB Test Set (3000 height)
    image_list = "/home/max/data/la/textblock_detection/newseye_tb_data/onb/tmp.lst"
    # Independance_lux dataset (3000 height)
    image_list = '/home/max/data/la/textblock_detection/bnl_data/independance_lux/traindata_headers/val.lst'

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
    # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/separator_detection/SEP_aru_5300/export/" \
    #              "SEP_aru_5300_2020-06-10.pb"
    tb_pp = TextBlockNetPostProcessor(image_list, path_to_pb, fixed_height=None, scaling_factor=0.55, threshold=0.05)
    # tb_pp = TextBlockNetPostProcessor2(path_to_pb, fixed_height=None, scaling_factor=0.7, threshold=0.1)
    tb_pp.run()
    # for i, image in enumerate(tb_pp.images):
    #     tb_pp.plot_binary(image, tb_pp.net_outputs[i][0, :, :, 0])

