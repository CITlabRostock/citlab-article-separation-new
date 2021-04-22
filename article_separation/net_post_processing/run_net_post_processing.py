import argparse
from concurrent.futures import ProcessPoolExecutor

from citlab_article_separation.net_post_processing.separator_net_post_processor import SeparatorNetPostProcessor
from citlab_article_separation.net_post_processing.heading_net_post_processor import HeadingNetPostProcessor
from citlab_python_util.io.file_loader import load_list_file


def run_separator(image_list, path_to_pb, fixed_height, scaling_factor, threshold):
    post_processor = SeparatorNetPostProcessor(image_list, path_to_pb, fixed_height, scaling_factor, threshold,
                                               gpu_devices='')
    post_processor.run()


def run_heading(image_list, path_to_pb, fixed_height=900, scaling_factor=1, is_heading_threshold=0.4, weight_dict=None,
                thresh_dict=None, text_line_percentage=0.8):
    if thresh_dict is None:
        thresh_dict = {'net_thresh': 1.0, 'stroke_width_thresh': 1.0, 'text_height_thresh': 0.9, 'sw_th_thresh': 0.9}
    if weight_dict is None:
        weight_dict = {'net': 0.8, 'stroke_width': 0.0, 'text_height': 0.2}
    post_processor = HeadingNetPostProcessor(image_list, path_to_pb, fixed_height, scaling_factor, weight_dict,
                                             is_heading_threshold, thresh_dict, text_line_percentage)
    post_processor.run(gpu_device='')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_image_list", type=str, required=True,
                        help="Path to the list file holding the image paths.")
    parser.add_argument("--path_to_pb", type=str, required=True,
                        help="Path to the TensorFlow pixel labelling graph.")
    parser.add_argument("--num_processes", type=int, required=False,
                        help="Number of processes that run in parallel.", default=8)
    parser.add_argument("--fixed_height", type=int, required=False,
                        help="Input image height")
    parser.add_argument("--scaling_factor", type=float, required=False,
                        help="Scaling factor of images.", default=1.0)
    parser.add_argument("--mode", type=str, required=True, choices=['heading', 'separator'],
                        help="Which information should be processed, e.g. headings or separator.")
    parser.add_argument("--threshold", type=float, required=False,
                        help="Threshold for binarization of net output.", default=0.05)

    args = parser.parse_args()

    mode = args.mode

    image_path_list = load_list_file(args.path_to_image_list)
    path_to_pb = args.path_to_pb
    num_processes = args.num_processes

    if args.fixed_height is None:
        if mode == 'heading':
            fixed_height = 900
        elif mode == 'separator':
            fixed_height = 1500
    else:
        fixed_height = args.fixed_height
    scaling_factor = args.scaling_factor
    threshold = args.threshold

    MAX_SUBLIST_SIZE = 50

    with ProcessPoolExecutor(num_processes) as executor:
        size_sub_lists = len(image_path_list) // num_processes
        if size_sub_lists == 0:
            size_sub_lists = 1
            num_processes = len(image_path_list)
        size_sub_lists = min(MAX_SUBLIST_SIZE, size_sub_lists)

        image_path_sub_lists = [image_path_list[i: i + size_sub_lists] for i in
                                range(0, len(image_path_list), size_sub_lists)]

        if mode == 'separator':
            run_args = ((image_path_sub_list, path_to_pb, fixed_height, scaling_factor, threshold) for
                        image_path_sub_list in image_path_sub_lists)

            [executor.submit(run_separator, *run_arg) for run_arg in run_args]
        elif mode == 'heading':
            run_args = ((image_path_sub_list, path_to_pb, fixed_height, scaling_factor, 0.4, None, None, 0.8)
                        for image_path_sub_list in image_path_sub_lists)

            [executor.submit(run_heading, *run_arg) for run_arg in run_args]
