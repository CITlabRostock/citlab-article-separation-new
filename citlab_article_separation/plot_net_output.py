from __future__ import print_function, division

import colorsys
import os
import random
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 255,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c],
                                  image[:, :, c])
    return image

def plot_image_with_net_output(image, net_output):
    colors = random_colors(10)
    print(colors)
    # print(net_output.shape)
    # net_output_rgb_int = np.uint8(cv2.cvtColor(net_output, cv2.COLOR_GRAY2RGB))
    # net_output_rgb_int = np.uint8(np.where(net_output_rgb_int == [255, 255, 255], [0, 255, 0], [0, 0, 0]))
    # print(net_output_rgb_int.shape)
    # # net_output_rgb_int = np.uint8(cv2.cvtColor(net_output, cv2.COLOR_GRAY2BGR))
    # # net_output_rgb_int = cv2.cvtColor(net_output_rgb_int, cv2.COLOR_BGR2HLS)
    #
    # res = cv2.addWeighted(image, 0.4, net_output_rgb_int, 0.6, 0)
    # # added_image = cv2.addWeighted(background, 0.4, overlay, 0.1, 0)
    # # cv2.addWeighted(net_output_rgb_int, 0.25, image, 0.75, 0, image)
    # # res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # # return image
    return apply_mask(image, net_output, (255, 50, 50), 0.5)


def plot_connected_components(image):
    _, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - image_bin)

    # res = cv2.addWeighted(image, 0.4, labels, 0.1, 0, dtype=2)
    return labels


def compute_accuracy(hyp_image, gt_image):
    """
    Compare the hypothesis image `hyp_image` to the GT image `gt_image` and compute the accuracy.
    Assume that both images are binary.
    :param hyp_image:
    :param gt_image:
    :return:
    """
    return np.sum(hyp_image == gt_image) / gt_image.size


def plot_confidence_histogram(bin_image):
    """ Assumes that `bin_image` is a binary image with values ranging from 0 to 255.

    :param bin_image:
    :return:
    """
    plt.hist(bin_image.flatten(), bins=256, range=(0, 1))
    plt.show()


def plot_net_output(path_to_pb, path_to_img_lst, save_folder="", gpu_device="0", rescale=None, fixed_height=None,
                    mask_threshold=None, plot_with_gt=False, plot_with_img=False, show_plot=False,
                    calculate_accuracy=True, figsize=(16, 16), ax=None):
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.visible_device_list = gpu_device

    graph = load_graph(path_to_pb)

    with tf.Session(graph=graph, config=session_conf) as sess:
        x = graph.get_tensor_by_name('inImg:0')
        out = graph.get_tensor_by_name('output:0')

        with open(path_to_img_lst) as f:
            accuracies = []
            for path_to_img in f.readlines():

                path_to_img = path_to_img.rstrip()

                dirname = os.path.dirname(path_to_img)
                img_name, ext = os.path.splitext(os.path.basename(path_to_img))

                img = cv2.imread(path_to_img)
                img_height, img_width = img.shape[:2]
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                scaling_factor = None
                if fixed_height and rescale and rescale != 1:
                    scaling_factor = rescale * fixed_height / img_height
                elif fixed_height:
                    scaling_factor = fixed_height / img_height
                elif rescale:
                    scaling_factor = rescale

                if 0.1 < scaling_factor < 1.0:
                    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                    img_gray = cv2.resize(img_gray, None, fx=scaling_factor, fy=scaling_factor,
                                          interpolation=cv2.INTER_AREA)

                if len(img_gray.shape) == 2:
                    img_gray = np.expand_dims(img_gray, axis=-1)
                    img_gray = np.expand_dims(img_gray, axis=0)

                img_gray = img_gray / 255.0
                out_img = sess.run(out, feed_dict={x: img_gray})

                n_class_img = out_img.shape[-1]

                print("Percentage of Pixels where the net is not 100% sure: ",
                      np.sum((0 < out_img) & (out_img < 1)) / out_img.size)

                if mask_threshold:
                    out_img = np.array((out_img > 0.6), np.int32)

                n_class_img_name = n_class_img
                # n_class_img_name = 2

                if plot_with_gt or calculate_accuracy:
                    paths_to_gts = [
                        os.path.join(dirname, "C" + str(n_class_img_name), img_name + "_GT" + str(i) + ".png")
                        for i in range(n_class_img_name)]
                    gt_imgs = [cv2.imread(path_to_gt) for path_to_gt in paths_to_gts]
                    gt_imgs = [cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) for gt_img in gt_imgs]
                    if scaling_factor:
                        gt_imgs = [cv2.resize(gt_img, None, fx=scaling_factor, fy=scaling_factor) for gt_img in gt_imgs]

                out_img_2d_argmax_values = np.argmax(out_img, axis=3)
                out_img_2d_argmax = np.zeros_like(out_img)
                # out_img_2d = np.where(out_img_2d_argmax_values, [])
                class_pixel_counts = {"class_" + str(i): 0 for i in range(n_class_img)}
                full_count = img_width * img_height
                for i in range(out_img.shape[0]):
                    for j in range(out_img.shape[1]):
                        for k in range(out_img.shape[2]):
                            argmax = out_img_2d_argmax_values[i, j, k]
                            out_img_2d_argmax[i, j, k, argmax] = 1
                            class_pixel_counts["class_" + str(argmax)] += 1
                for class_name, class_pixel_count in class_pixel_counts.items():
                    print(f"Percentage of pixels in {class_name}: {class_pixel_count / full_count}")

                accuracy = 0
                for cl in range(n_class_img_name):
                    out_img_2d = out_img[0, :, :, cl]
                    out_img_2d_255 = out_img_2d * 255
                    out_img_2d_255 = np.uint8(out_img_2d_255)

                    # Make histogram plot_binary
                    # plot_confidence_histogram(out_img_2d_255)

                    # calculate accuracy
                    if calculate_accuracy:
                        accuracy += compute_accuracy(out_img_2d_argmax[0, :, :, cl], gt_imgs[cl] / 255)

                    if plot_with_img:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        out_img_2d_255 = plot_image_with_net_output(img.astype(np.uint32), out_img_2d_255)
                    if plot_with_gt:
                        gt_img = gt_imgs[cl]
                        if plot_with_img:
                            gt_img = plot_image_with_net_output(img, gt_img)
                        out_img_2d_255 = np.concatenate((out_img_2d_255, gt_img), axis=1)
                    if save_folder:
                        cv2.imwrite(
                            os.path.join(save_folder, img_name + "_OUT" + str(cl) + ext), cv2.cvtColor(np.uint8(out_img_2d_255), cv2.COLOR_RGB2BGR))
                    if show_plot:
                        if plot_with_img:
                            _, ax = plt.subplots(1, figsize=figsize)

                            # Show area outside image boundaries.
                            height, width = img.shape[:2]
                            ax.set_ylim(height + 10, -10)
                            ax.set_xlim(-10, width + 10)
                            ax.axis('off')

                            ax.imshow(out_img_2d_255.astype(np.uint8))
                        else:
                            plt.imshow(out_img_2d_255, cmap="gray")
                        # img_gray_2d = img_gray[0, :, :, 0]
                        # #
                        # # img_with_net_output = plot_image_with_net_output(img, out_img_2d_255)
                        # # plt.imshow(img_with_net_output, cmap='gray')
                        # # plt.imshow(img)
                        # # print(img.shape)
                        # # plt.show()
                        # # plt.imshow(img_with_net_output)
                        # img_with_ccs = plot_connected_components(img_gray_2d)
                        # plt.imshow(img_with_ccs, cmap='gray')
                        plt.show()
                        # exit(1)
                accuracy /= n_class_img
                accuracies.append(accuracy)
                print("Accuracy = ", accuracy)
                print("+++++++++++++++++++++++")
            print("Overall Accuracy = ", sum(accuracies) / len(accuracies))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_to_tf_graph', type=str,
                        help="path to the TensorFlow .pb file containing the ARU-Net graph.")
    parser.add_argument('--path_to_img_lst', type=str,
                        help='path to the lst file containing the file paths of the images.')
    parser.add_argument('--save_folder', type=str, help="path to the save folder")
    parser.add_argument('--rescale_factor', default=1.0, type=float,
                        help="rescaling the images before inputting them to the network.")
    parser.add_argument('--fixed_height', default=0, type=int,
                        help="rescales the input images to a fixed height. Only works if rescale_factor is not set")
    parser.add_argument('--calculate_accuracy', default=False, action='store_true',
                        help="whether or not to calculate the accuracy of the net output - needs GT.")

    args = parser.parse_args()

    path_to_tf_graph = args.path_to_tf_graph
    path_to_img_lst = args.path_to_img_lst
    save_folder = args.save_folder

    path_to_tf_graph = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
                       "racetrack_onb_textblock_136/TB_aru_1250_height/export/TB_aru_1250_height_2020-05-16.pb"

    # path_to_tf_graph = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #                    "racetrack_onb_textblock_136/TB_test_accuracy_measure/export/TB_test_accuracy_measure_2020-05-20.pb"

    # path_to_tf_graph = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/independance_lux/" \
    #                    "headers/tb_headers_aru/export/tb_headers_aru_2020-05-27.pb"
    # path_to_img_lst = "/home/max/data/la/textblock_detection/bnl_data/independance_lux/traindata_headers/val.lst"

    # if not os.path.exists(args.save_folder) or not os.path.isdir(args.save_folder) and args.save_folder:
    #     os.mkdir(args.save_folder)

    path_to_tf_graph = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
                       "racetrack_onb_textblock_136/TB_aru_3000_height_scaling_train_only/export/" \
                       "TB_aru_3000_height_scaling_train_only_2020-06-05.pb"

    path_to_tf_graph = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
                       "racetrack_onb_textblock_136/TB_aru_3000_height_scaling_train_only/export/" \
                       "TB_aru_3000_height_scaling_train_only_2020-06-05.pb"

    # with outlines

    # path_to_tf_graph = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #                    "racetrack_onb_textblock_136/TB_aru_3000_outlines_height_scaling_train_only/export/" \
    #                    "TB_aru_3000_outlines_height_scaling_train_only_2020-06-05.pb"

    # with separator class
    # path_to_tf_graph = "/home/max/devel/projects/python/article_separation/models/" \
    #                    "AS_newspaper_onb_textblock_aru_dist_020_to_050scale_4classes_150epochs/" \
    #                    "export/AS_newspaper_onb_textblock_aru_dist_020_to_050scale_4classes_2019-10-23.pb"

    # headers
    # path_to_tf_graph = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/independance_lux/" \
    #                    "headers/tb_headers_aru/export/tb_headers_aru_2020-06-04.pb"

    plot_net_output(path_to_tf_graph, path_to_img_lst, save_folder, rescale=args.rescale_factor,
                    fixed_height=1500, plot_with_gt=False, plot_with_img=True,
                    mask_threshold=True, show_plot=True, calculate_accuracy=False)
