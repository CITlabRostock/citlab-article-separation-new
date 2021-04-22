import os

import cv2
import numpy as np
import tensorflow as tf

from citlab_python_util.basic import shape_utils
from citlab_python_util.io.file_loader import load_image
from citlab_python_util.io.file_loader import load_list_file
from citlab_python_util.logging import custom_logging

logger = custom_logging.setup_custom_logger("ImageResizer", "info")


class ImageResizer:
    def __init__(self, path_to_image_list, max_height=0, max_width=0, scaling_factor=1.0):
        self.image_path_list = load_list_file(path_to_image_list)
        self.image_list = [load_image(image_path, "pil", mode="L") for image_path in self.image_path_list]
        self.image_resolutions = [pil_image.size for pil_image in self.image_list]  # (width, height) resolutions
        self.max_width = max(0, max_width)
        self.max_height = max(0, max_height)
        if (self.max_height, self.max_width) is not (0, 0):
            self.scaling_factors = self.calculate_scaling_factors_from_max_resolution()
        else:
            self.scaling_factors = [scaling_factor] * len(self.image_list)
        self.resized_images = []

    def set_max_resolution(self, max_height=0, max_width=0):
        self.max_height = max_height
        self.max_width = max_width
        self.scaling_factors = self.calculate_scaling_factors_from_max_resolution()

    def save_resized_images(self, save_folder):
        if len(self.resized_images) == 0:
            self.resize_images()

        for i, (image_path, resized_image) in enumerate(zip(self.image_path_list, self.resized_images)):
            image_name = os.path.basename(image_path)
            save_path = os.path.join(save_folder, image_name)

            logger.debug(f"Save image in {save_path}")
            logger.debug(f"Scaling factor: {self.scaling_factors[i]}")
            logger.debug(f"Max_height: {self.max_height}")
            logger.debug(f"Max_width: {self.max_width}")

            cv2.imwrite(save_path, resized_image)

    def calculate_scaling_factors_from_max_resolution(self):
        if (self.max_height, self.max_width) == (0, 0):
            logger.debug("No max resolution given, do nothing...")
            return [1.0] * len(self.image_path_list)

        if self.max_height == 0:
            return [min(1.0, self.max_width / img_res[0]) for img_res in self.image_resolutions]
        elif self.max_width == 0:
            return [min(1.0, self.max_height / img_res[1]) for img_res in self.image_resolutions]
        else:
            return [min(1.0, max(self.max_width / img_res[0], self.max_height / img_res[1])) for img_res
                    in self.image_resolutions]

    def resize_images(self):
        self.resized_images = [self.resize_image(pil_image, sc) for pil_image, sc in
                               zip(self.image_list, self.scaling_factors)]

    def resize_image(self, pil_image, scaling_factor):
        image = np.array(pil_image, np.uint8)
        if scaling_factor < 1:
            return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
        elif scaling_factor > 1:
            return cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_CUBIC)
        else:
            return image


def resize_image_fixed(tensor_dict,
                       new_height=600,
                       new_width=1024,
                       method=tf.image.ResizeMethod.BILINEAR,
                       align_corners=False):
    """Resizes images to the given height and width and adds the new image shapes to the tensor dict.

    Args:
      tensor_dict:
      new_height: (optional) (scalar) desired height of the image.
      new_width: (optional) (scalar) desired width of the image.
      method: (optional) interpolation method used in resizing. Defaults to
              BILINEAR.
      align_corners: bool. If true, exactly align all 4 corners of the input
                     and output. Defaults to False.

    Returns:
      Note that the position of the resized_image_shape changes based on whether
      masks are present.
      resized_image: A tensor of size [new_height, new_width, channels].
      resized_masks: If masks is not None, also outputs masks. A 3D tensor of
        shape [num_instances, new_height, new_width]
      resized_image_shape: A 1D tensor of shape [3] containing the shape of the
        resized image.
    """
    with tf.name_scope(
            'ResizeImage',
            values=[tensor_dict['image'], new_height, new_width, method, align_corners]):
        tensor_dict['image'] = tf.image.resize(
            tensor_dict['image'], tf.stack([new_height, new_width]),
            method=method,
            align_corners=align_corners)
        tensor_dict['image_shape'] = shape_utils.combined_static_and_dynamic_shape(tensor_dict['image'])
        return tensor_dict


def resize_image_ratio(tensor_dict,
                       min_dimension=600,
                       max_dimension=1024,
                       method=tf.image.ResizeMethod.BILINEAR,
                       align_corners=False,
                       pad_to_max_dimension=False,
                       per_channel_pad_value=(0, 0, 0)):
    """Resizes an image so its dimensions are within the provided values.

    Args:
      tensor_dict:
      min_dimension: (scalar) desired size of the smaller image
                     dimension.
      max_dimension: (scalar) maximum allowed size
                     of the larger image dimension.
      method: (optional) interpolation method used in resizing. Defaults to
              BILINEAR.
      align_corners: bool. If true, exactly align all 4 corners of the input
                     and output. Defaults to False.
      pad_to_max_dimension: Whether to resize the image and pad it with zeros
        so the resulting image is of the spatial size
        [max_dimension, max_dimension]. If masks are included they are padded
        similarly.
      per_channel_pad_value: A tuple of per-channel scalar value to use for
        padding. By default pads zeros.

    Returns:
        tensor_dict

    """
    with tf.name_scope('ResizeToRange', values=[tensor_dict['image'], min_dimension]):
        image = tensor_dict['image']
        if image.get_shape().is_fully_defined():
            new_size = _compute_new_static_size(image, min_dimension, max_dimension)
        else:
            new_size = _compute_new_dynamic_size(image, min_dimension, max_dimension)
        tensor_dict['image_shape'] = new_size
        new_image = tf.image.resize(
            image, new_size[:-1], method=method, align_corners=align_corners)
        tensor_dict['image'] = new_image
        if pad_to_max_dimension:
            channels = tf.unstack(new_image, axis=2)
            if len(channels) > len(per_channel_pad_value):
                raise ValueError('Number of channels must be less or equal to the length of '
                                 'per-channel pad value.')
            new_image = tf.stack(
                [
                    tf.pad(
                        channels[i], [[0, max_dimension - new_size[0]],
                                      [0, max_dimension - new_size[1]]],
                        constant_values=per_channel_pad_value[i])
                    for i in range(len(channels))
                ],
                axis=2)
            new_image.set_shape([max_dimension, max_dimension, len(channels)])

        return tensor_dict


def _compute_new_static_size(image, min_dimension, max_dimension):
    """Compute new static shape for resize_to_range method."""
    image_shape = image.get_shape().as_list()
    orig_height = image_shape[0]
    orig_width = image_shape[1]
    num_channels = image_shape[2]

    # Scale factor such that maximal dimension is at most max_dimension
    orig_max_dim = max(orig_height, orig_width)
    small_scale_factor = max_dimension / float(orig_max_dim)
    # if this factor is less than 1 we have to act!

    # Scale factor such that minimal dimension is at least min_dimension
    orig_min_dim = min(orig_height, orig_width)
    large_scale_factor = min_dimension / float(orig_min_dim)
    # If image is already big enough... do nothing
    large_scale_factor = max(large_scale_factor, 1.0)

    # Take the minimum (we ensure that maxdim is not exceeded and if possible min_dim is met also)
    scale_factor = min(small_scale_factor, large_scale_factor)

    new_height = int(round(orig_height * scale_factor))
    new_width = int(round(orig_width * scale_factor))
    new_size = [new_height, new_width]
    return tf.constant(new_size + [num_channels])


def _compute_new_dynamic_size(image, min_dimension, max_dimension):
    """Compute new dynamic shape for resize_to_range method."""
    image_shape = tf.shape(image)
    orig_height = tf.cast(image_shape[0], dtype=tf.float32)
    orig_width = tf.cast(image_shape[1], dtype=tf.float32)
    num_channels = image_shape[2]

    # Scale factor such that maximal dimension is at most max_dimension
    orig_max_dim = tf.maximum(orig_height, orig_width)
    max_dimension = tf.constant(max_dimension, dtype=tf.float32)
    small_scale_factor = max_dimension / orig_max_dim
    # if this factor is less than 1 we have to act!

    # Scale factor such that minimal dimension is at least min_dimension
    orig_min_dim = tf.minimum(orig_height, orig_width)
    min_dimension = tf.constant(min_dimension, dtype=tf.float32)
    large_scale_factor = min_dimension / orig_min_dim
    # If image is already big enough... do nothing
    large_scale_factor = tf.maximum(large_scale_factor, 1.0)

    # Take the minimum (we ensure that maxdim is not exceeded and if possible min_dim is met also)
    scale_factor = tf.minimum(small_scale_factor, large_scale_factor)

    new_height = tf.cast(tf.round(orig_height * scale_factor), dtype=tf.int32)
    new_width = tf.cast(tf.round(orig_width * scale_factor), dtype=tf.int32)
    new_size = tf.stack([new_height, new_width])
    return tf.stack(tf.unstack(new_size) + [num_channels])


if __name__ == '__main__':
    max_heights = [i for i in range(1000, 3000, 500)]
    image_resizer = ImageResizer(
        path_to_image_list="/home/max/data/la/racetrack_onb_corrected_baselines_no_tifs/racetrack_onb_corrected_baselines.lst")
    for max_height in max_heights:
        image_resizer.set_max_resolution(max_height=max_height)
        save_folder = save_folder = "/home/max/newspaper_different_heights/" + str(max_height)
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        image_resizer.resize_images()
        image_resizer.save_resized_images(save_folder)
