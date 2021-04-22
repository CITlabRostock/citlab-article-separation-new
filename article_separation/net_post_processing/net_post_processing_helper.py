import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from citlab_python_util.image_processing.image_stats import get_scaling_factor


def load_image_paths(image_list):
    with open(image_list) as f:
        image_paths = f.readlines()
    return [image_path.rstrip() for image_path in image_paths]


def scale_image(image, fixed_height=None, scaling_factor=1.0):
    # image_width, image_height = image.shape[:2]
    image_height, image_width = image.shape[:2]

    sc = get_scaling_factor(image_height, image_width, scaling_factor, fixed_height=fixed_height)
    if sc < 1.0:
        image = cv2.resize(image, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    elif sc > 1.0:
        # if INTER_CUBIC is too slow try INTER_LINEAR
        image = cv2.resize(image, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)

    return image, sc


def load_and_scale_image(path_to_image, fixed_height, scaling_factor):
    image = cv2.imread(path_to_image)
    image, sc = scale_image(image, fixed_height, scaling_factor)
    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) / 255.0

    return image, image_grey, sc


def load_graph(path_to_pb):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(path_to_pb, "rb") as f:
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


def get_net_output(image, pb_graph: tf.Graph, gpu_device="0"):
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)

    session_conf = tf.ConfigProto()
    session_conf.gpu_options.visible_device_list = gpu_device

    if gpu_device == "" or gpu_device is None:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with tf.Session(graph=pb_graph, config=session_conf) as sess:
        x = sess.graph.get_tensor_by_name('inImg:0')
        out = sess.graph.get_tensor_by_name('output:0')

        return sess.run(out, feed_dict={x: image})[0]


def apply_threshold(net_output, threshold):
    if net_output.dtype == np.uint8:
        threshold *= 255
    return np.array((net_output > threshold) * 255, dtype=np.uint8)


def _plot_image_with_net_output(image, net_output):
    net_output_rgb_int = np.uint8(cv2.cvtColor(net_output, cv2.COLOR_GRAY2BGR))
    net_output_rgb_int = cv2.cvtColor(net_output_rgb_int, cv2.COLOR_BGR2HLS)

    res = cv2.addWeighted(image, 0.9, net_output_rgb_int, 0.4, 0)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res


def plot_net_output(net_output, image=None):
    num_classes = net_output.shape[-1]
    for cl in range(num_classes):
        net_output_2d = net_output[0, :, :, cl]
        net_output_2d = net_output_2d * 255
        net_output_2d = np.uint8(net_output_2d)

        if image is not None:
            image_plot = _plot_image_with_net_output(image, net_output_2d)
            plt.imshow(image_plot)
        else:
            image_plot = net_output_2d
            plt.imshow(image_plot, cmap="gray")
        plt.show()


