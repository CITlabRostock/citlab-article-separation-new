"""
This file is used for evaluating different hyperparameter settings for the heading detection performed by the
```HeadingNetPostProcessor```. One specific hyperparameter setting is run with the ```heading_evaluation``` method.
"""

import concurrent.futures
import os
import argparse


def run_grid_search(fixed_height, threshold, net_weight, net_thresh, stroke_width_thresh, text_height_thresh,
                    text_line_percentage):
    """
    Run a grid search on different hyperparameters to find the best setting for detecting headings on a page.
    :param fixed_height: All images should be scaled to have this fixed height.
    :type fixed_height: int
    :param threshold: A value in [0, 1] that should be reached by the heading detection confidence value for a specific
    text line to define it as a heading text line.
    :type threshold: float
    :param net_weight: A value in [0, 10] that weights the confidence value of the neural network output. Gets
    downscaled to [0, 1] later on.
    :type net_weight: int
    :param net_thresh: If the net confidence is greater than or equal to this value the text line is considered a
    heading. Gets downscaled to [0, 1] later on.
    :type net_thresh: int
    :param stroke_width_thresh: If the confidence of the stroke width feature coming from the distance transformation
    is greater than or equal to this value the text line is considered a heading. Gets downscaled to [0, 1] later on.
    :type stroke_width_thresh: int
    :param text_height_thresh: If the confidence of the text height feature coming from the distance transformation
    is greater than or equal to this value the text line is considered a heading. Gets downscaled to [0, 1] later on.
    :type text_height_thresh: int
    :param text_line_percentage: For a TextRegion to be defined as a heading region at least this percentage of text
    lines should be recognized as headings by the algorithm. Defined as a value in [0, 10]. Gets downscaled to [0, 1]
    later on.
    :type text_line_percentage: int
    :return:
    """
    net_weight_f = net_weight / 10
    net_thresh_f = net_thresh / 10
    stroke_width_thresh_f = stroke_width_thresh / 10
    text_height_thresh_f = text_height_thresh / 10
    text_line_percentage_f = text_line_percentage / 10

    sw_th_thresh_upper_bound = min(stroke_width_thresh, text_height_thresh)
    for sw_th_thresh in range(sw_th_thresh_upper_bound - 1, sw_th_thresh_upper_bound + 1, 1):
        sw_th_thresh_f = sw_th_thresh / 10

        for stroke_width_weight in range(0, 10 - net_weight + 1, 1):
            stroke_width_weight_f = stroke_width_weight / 10
            text_height_weight_f = (10 - net_weight - stroke_width_weight) / 10

            os.system("python -u "
                      "./citlab-article-separation/article_separation/image_segmentation/net_post_processing/heading_evaluation.py "
                      "--path_to_gt_list {} "
                      "--path_to_pb {} "
                      "--fixed_height {} "
                      "--threshold {} "
                      "--net_weight {} "
                      "--stroke_width_weight {} "
                      "--text_height_weight {} "
                      "--gpu_devices '' "
                      "--log_file_folder {} "
                      "--net_thresh {} "
                      "--stroke_width_thresh {} "
                      "--text_height_thresh {} "
                      "--sw_th_thresh {} "
                      "--text_line_percentage {}"
                      .format(PATH_TO_GT_LIST, PATH_TO_PB, fixed_height, threshold, net_weight_f, stroke_width_weight_f,
                              text_height_weight_f, LOG_FILE_FOLDER, net_thresh_f, stroke_width_thresh_f,
                              text_height_thresh_f, sw_th_thresh_f, text_line_percentage_f))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_gt_list", type=str, required=True,
                        help='Path to GT image list',
                        default='/home/max/data/la/heading_detection/post_process_experiments/image_paths_dummy.lst')
    parser.add_argument("--path_to_pb", type=str, required=True,
                        help='Path to the TensorFlow graph.',
                        default='/home/max/data/la/heading_detection/post_process_experiments/HD_ru_3000_export_best_2020-09-22.pb')
    parser.add_argument("--log_file_folder", type=str, required=True,
                        help='Path to the folder where the log files are stored',
                        default='/home/max/tests/logs')
    parser.add_argument("--num_processes", type=int, required=False,
                        help='Number of parallel processes.', default=8)

    cmd_args = parser.parse_args()
    PATH_TO_GT_LIST = cmd_args.path_to_gt_list
    PATH_TO_PB = cmd_args.path_to_pb
    LOG_FILE_FOLDER = cmd_args.log_file_folder
    num_processes = cmd_args.num_processes

    with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
        fixed_heights = range(600, 1300, 100)
        thresholds = range(4, 10, 1)
        net_weights = range(0, 11, 1)

        net_threshs = range(8, 11, 1)
        stroke_width_threshs = range(8, 11, 1)
        text_height_threshs = range(8, 11, 1)
        text_line_percentage = range(8, 11, 1)

        args = ((f, t/10, nw, nt, swt, tht, tlp) for f in fixed_heights for t in thresholds for nw in net_weights
                for nt in net_threshs for swt in stroke_width_threshs for tht in text_height_threshs
                for tlp in text_line_percentage)
        [executor.submit(run_grid_search, *arg) for arg in args]


