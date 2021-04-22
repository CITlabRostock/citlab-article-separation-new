import argparse
import os

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

from citlab_article_separation.net_post_processing.heading_net_post_processor import HeadingNetPostProcessor
from citlab_python_util.io.file_loader import load_list_file, get_page_path
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.page_constants import TextRegionTypes


def get_heading_regions(page_object: Page):
    text_regions = page_object.get_text_regions()
    return [text_region for text_region in text_regions if text_region.region_type == TextRegionTypes.sHEADING]


def get_heading_text_lines(heading_regions):
    text_lines = []
    for heading_region in heading_regions:
        text_lines.extend(heading_region.text_lines)
    return text_lines


def get_heading_text_line_by_custom_type(heading_regions):
    text_lines = []
    for heading_region in heading_regions:
        from citlab_python_util.parser.xml.page.page_objects import TextLine
        text_line: TextLine
        for text_line in heading_region.text_lines:
            try:
                if text_line.custom["structure"]["semantic_type"] == TextRegionTypes.sHEADING:
                    text_lines.append(text_line)
            except KeyError:
                continue

    return text_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_gt_list', type=str, required=True,
                        help='Path to the list of GT PAGE XML file paths.')
    parser.add_argument('--path_to_pb', type=str, required=True,
                        help="Path to the TensorFlow pb graph for creating the separator information")
    parser.add_argument('--fixed_height', type=int, required=False,
                        help="If parameter is given, the images will be scaled to this height by keeping the aspect "
                             "ratio")
    parser.add_argument('--threshold', type=float, required=False,
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
    parser.add_argument('--log_file_folder', type=str, required=False, help='Where to store the log files.')

    args = parser.parse_args()

    path_to_gt_list = args.path_to_gt_list
    path_to_pb = args.path_to_pb
    fixed_height = args.fixed_height
    is_heading_threshold = args.threshold
    net_weight = args.net_weight
    stroke_width_weight = args.stroke_width_weight
    text_height_weight = args.text_height_weight
    net_thresh = args.net_thresh
    stroke_width_thresh = args.stroke_width_thresh
    text_height_thresh = args.text_height_thresh
    sw_th_thresh = args.sw_th_thresh
    text_line_percentage = args.text_line_percentage
    log_file_folder = args.log_file_folder
    gpu_devices = args.gpu_devices

    # net_weight = 0.33
    # stroke_width_weight = 0.33
    # text_height_weight = 0.33

    weight_dict = {"net": net_weight,
                   "stroke_width": stroke_width_weight,
                   "text_height": text_height_weight}

    thresh_dict = {"net_thresh": net_thresh,
                   "stroke_width_thresh": stroke_width_thresh,
                   "text_height_thresh": text_height_thresh,
                   "sw_th_thresh": sw_th_thresh}


    # path_to_gt_list = "/home/max/data/la/heading_detection/post_process_experiments/image_paths.lst"
    #
    # path_to_gt_list = '/home/max/data/la/heading_detection/post_process_experiments/dummy_image_paths.lst'
    # path_to_pb = "/home/max/data/la/heading_detection/post_process_experiments/HD_ru_3000_export_best_2020-09-22.pb"
    # fixed_height = 500
    # is_heading_threshold = 0.5

    image_paths = load_list_file(path_to_gt_list)

    post_processor = HeadingNetPostProcessor(path_to_gt_list, path_to_pb, fixed_height, scaling_factor=None,
                                             weight_dict=weight_dict, threshold=is_heading_threshold,
                                             thresh_dict=thresh_dict, text_line_percentage=text_line_percentage)
    page_objects_hyp = post_processor.run(gpu_devices)

    log_file_name = f"{fixed_height:04}_{is_heading_threshold*100:03.0f}_{net_weight*100:03.0f}_" \
                    f"{stroke_width_weight*100:03.0f}_{text_height_weight*100:03.0f}_" \
                    f"{net_thresh*100:03.0f}_{stroke_width_thresh*100:03.0f}_{text_height_thresh*100:03.0f}_" \
                    f"{text_line_percentage*100:03.0f}.log"
    log_file_name = os.path.join(log_file_folder, log_file_name)

    f1_scores_bin, recall_scores_bin, precision_scores_bin = [], [], []
    f1_scores_micro, recall_scores_micro, precision_scores_micro = [], [], []
    f1_scores_macro, recall_scores_macro, precision_scores_macro = [], [], []
    f1_scores_weighted, recall_scores_weighted, precision_scores_weighted = [], [], []

    with open(log_file_name, 'w') as log_file:
        log_file.write(f"fixed_height: {fixed_height}\n"
                       f"is_heading_threshold: {is_heading_threshold}\n"
                       f"net_weight: {net_weight}\n"
                       f"stroke_width_weight: {stroke_width_weight}\n"
                       f"text_height_weight: {text_height_weight}\n"
                       f"net_thresh: {net_thresh}\n"
                       f"stroke_width_thresh: {stroke_width_thresh}\n"
                       f"text_height_thresh: {text_height_thresh}\n"
                       f"sw_th_thresh: {sw_th_thresh}\n"
                       f"text_line_percentage: {text_line_percentage}\n")
        for i, image_path in enumerate(image_paths):
            log_file.write(f"\nImage path: {image_path}\n")
            xml_path = get_page_path(image_path)
            page_object_gt = Page(xml_path)
            page_object_hyp = page_objects_hyp[i]

            text_regions_gt = page_object_gt.get_text_regions()
            text_regions_hyp = page_object_hyp.get_text_regions()

            is_heading_gt = [tr.region_type == TextRegionTypes.sHEADING for tr in text_regions_gt]
            is_heading_hyp = [tr.region_type == TextRegionTypes.sHEADING for tr in text_regions_hyp]

            # Evaluation only on heading class
            recall_scores_bin.append(recall_score(is_heading_gt, is_heading_hyp, average='binary', zero_division=0))
            precision_scores_bin.append(precision_score(is_heading_gt, is_heading_hyp, average='binary', zero_division=0))
            f1_scores_bin.append(f1_score(is_heading_gt, is_heading_hyp, average='binary', zero_division=0))

            # Evaluation on heading and non-heading class
            recall_scores_micro.append(recall_score(is_heading_gt, is_heading_hyp, average='micro', zero_division=0))
            precision_scores_micro.append(precision_score(is_heading_gt, is_heading_hyp, average='micro', zero_division=0))
            f1_scores_micro.append(f1_score(is_heading_gt, is_heading_hyp, average='micro', zero_division=0))

            # Evaluation on heading and non-heading class separately (bin-case) and average the values
            recall_scores_macro.append(recall_score(is_heading_gt, is_heading_hyp, average='macro', zero_division=0))
            precision_scores_macro.append(precision_score(is_heading_gt, is_heading_hyp, average='macro', zero_division=0))
            f1_scores_macro.append(f1_score(is_heading_gt, is_heading_hyp, average='macro', zero_division=0))

            # Evaluation on heading and non-heading class
            # (weighted by support of each class, i.e. number of instances per class are taken into account)
            recall_scores_weighted.append(recall_score(is_heading_gt, is_heading_hyp, average='weighted', zero_division=0))
            precision_scores_weighted.append(precision_score(is_heading_gt, is_heading_hyp, average='weighted', zero_division=0))
            f1_scores_weighted.append(f1_score(is_heading_gt, is_heading_hyp, average='weighted', zero_division=0))

            log_file.write(f"\t{'R_BIN':>6}: {recall_scores_bin[-1]:.4f}")
            log_file.write(f"\t{'R_MIC':>6}: {recall_scores_micro[-1]:.4f}")
            log_file.write(f"\t{'R_MAC':>6}: {recall_scores_macro[-1]:.4f}")
            log_file.write(f"\t{'R_WEI':>6}: {recall_scores_weighted[-1]:.4f}\n")

            log_file.write(f"\t{'P_BIN':>6}: {precision_scores_bin[-1]:.4f}")
            log_file.write(f"\t{'P_MIC':>6}: {precision_scores_micro[-1]:.4f}")
            log_file.write(f"\t{'P_MAC':>6}: {precision_scores_macro[-1]:.4f}")
            log_file.write(f"\t{'P_WEI':>6}: {precision_scores_weighted[-1]:.4f}\n")

            log_file.write(f"\t{'F1_BIN':>6}: {f1_scores_bin[-1]:.4f}")
            log_file.write(f"\t{'F1_MIC':>6}: {f1_scores_micro[-1]:.4f}")
            log_file.write(f"\t{'F1_MAC':>6}: {f1_scores_macro[-1]:.4f}")
            log_file.write(f"\t{'F1_WEI':>6}: {f1_scores_weighted[-1]:.4f}")

        avg_recall_bin = np.mean(recall_scores_bin)
        avg_precision_bin = np.mean(precision_scores_bin)
        avg_f1_bin = np.mean(f1_scores_bin)

        avg_recall_micro = np.mean(recall_scores_micro)
        avg_precision_micro = np.mean(precision_scores_micro)
        avg_f1_micro = np.mean(f1_scores_micro)

        avg_recall_macro = np.mean(recall_scores_macro)
        avg_precision_macro = np.mean(precision_scores_macro)
        avg_f1_macro = np.mean(f1_scores_macro)

        avg_recall_weighted = np.mean(recall_scores_weighted)
        avg_precision_weighted = np.mean(precision_scores_weighted)
        avg_f1_weighted = np.mean(f1_scores_weighted)

        log_file.write("\n\nAverage Recall (BIN) \t Average Precision (BIN) \t Average F1 (BIN)\n")
        log_file.write(f"{avg_recall_bin:.4f}, {avg_precision_bin:.4f}, {avg_f1_bin:.4f}\n\n")
        log_file.write("\nAverage Recall (MIC) \t Average Precision (MIC) \t Average F1 (MIC)\n")
        log_file.write(f"{avg_recall_micro:.4f}, {avg_precision_micro:.4f}, {avg_f1_micro:.4f}\n\n")
        log_file.write("\nAverage Recall (MAC) \t Average Precision (MAC) \t Average F1 (MAC)\n")
        log_file.write(f"{avg_recall_macro:.4f}, {avg_precision_macro:.4f}, {avg_f1_macro:.4f}\n\n")
        log_file.write("\nAverage Recall (WEI) \t Average Precision (WEI) \t Average F1 (WEI)\n")
        log_file.write(f"{avg_recall_weighted:.4f}, {avg_precision_weighted:.4f}, {avg_f1_weighted:.4f}")
