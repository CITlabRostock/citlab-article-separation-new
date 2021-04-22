# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Utils used to manipulate tensor shapes."""

import tensorflow as tf


def get_feature_map_spatial_dims(feature_maps):
    """Return list of spatial dimensions for each feature map in a list.

    Args:
      feature_maps: a list of tensors where the ith tensor has shape
          [batch, height_i, width_i, depth_i].

    Returns:
      a list of pairs (height, width) for each feature map in feature_maps
    """
    feature_map_shapes = [
        combined_static_and_dynamic_shape(
            feature_map) for feature_map in feature_maps
    ]
    return [(shape[1], shape[2]) for shape in feature_map_shapes]


def fixed_padding(inputs, kernel_size, rate=1):
    """Pads the input along the spatial dimensions independently of input size.

    Args:
      inputs: A tensor of size [batch, height_in, width_in, channels].
      kernel_size: The kernel to be used in the conv2d or max_pool2d operation.
                   Should be a positive integer.
      rate: An integer, rate for atrous convolution.

    Returns:
      output: A tensor of size [batch, height_out, width_out, channels] with the
        input, either intact (if kernel_size == 1) or padded (if kernel_size > 1).
    """
    kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
    pad_total = kernel_size_effective - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs


def static_or_dynamic_map_fn(fn, elems, dtype=None,
                             parallel_iterations=32, back_prop=True):
    """Runs map_fn as a (static) for loop when possible.

    This function rewrites the map_fn as an explicit unstack input -> for loop
    over function calls -> stack result combination.  This allows our graphs to
    be acyclic when the batch size is static.
    For comparison, see https://www.tensorflow.org/api_docs/python/tf/map_fn.

    Note that `static_or_dynamic_map_fn` currently is not *fully* interchangeable
    with the default tf.map_fn function as it does not accept nested inputs (only
    Tensors or lists of Tensors).  Likewise, the output of `fn` can only be a
    Tensor or list of Tensors.

    TODO(jonathanhuang): make this function fully interchangeable with tf.map_fn.

    Args:
      fn: The callable to be performed. It accepts one argument, which will have
        the same structure as elems. Its output must have the
        same structure as elems.
      elems: A tensor or list of tensors, each of which will
        be unpacked along their first dimension. The sequence of the
        resulting slices will be applied to fn.
      dtype:  (optional) The output type(s) of fn. If fn returns a structure of
        Tensors differing from the structure of elems, then dtype is not optional
        and must have the same structure as the output of fn.
      parallel_iterations: (optional) number of batch items to process in
        parallel.  This flag is only used if the native tf.map_fn is used
        and defaults to 32 instead of 10 (unlike the standard tf.map_fn default).
      back_prop: (optional) True enables support for back propagation.
        This flag is only used if the native tf.map_fn is used.

    Returns:
      A tensor or sequence of tensors. Each tensor packs the
      results of applying fn to tensors unpacked from elems along the first
      dimension, from first to last.
    Raises:
      ValueError: if `elems` a Tensor or a list of Tensors.
      ValueError: if `fn` does not return a Tensor or list of Tensors
    """
    if isinstance(elems, list):
        for elem in elems:
            if not isinstance(elem, tf.Tensor):
                raise ValueError('`elems` must be a Tensor or list of Tensors.')

        elem_shapes = [elem.shape.as_list() for elem in elems]
        # Fall back on tf.map_fn if shapes of each entry of `elems` are None or fail
        # to all be the same size along the batch dimension.
        for elem_shape in elem_shapes:
            if (not elem_shape or not elem_shape[0]
                    or elem_shape[0] != elem_shapes[0][0]):
                return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
        arg_tuples = zip(*[tf.unstack(elem) for elem in elems])
        outputs = [fn(arg_tuple) for arg_tuple in arg_tuples]
    else:
        if not isinstance(elems, tf.Tensor):
            raise ValueError('`elems` must be a Tensor or list of Tensors.')
        elems_shape = elems.shape.as_list()
        if not elems_shape or not elems_shape[0]:
            return tf.map_fn(fn, elems, dtype, parallel_iterations, back_prop)
        outputs = [fn(arg) for arg in tf.unstack(elems)]
    # Stack `outputs`, which is a list of Tensors or list of lists of Tensors
    if all([isinstance(output, tf.Tensor) for output in outputs]):
        return tf.stack(outputs)
    else:
        if all([isinstance(output, list) for output in outputs]):
            if all([all([isinstance(entry, tf.Tensor) for entry in output_list]) for output_list in outputs]):
                return [tf.stack(output_tuple) for output_tuple in zip(*outputs)]
    raise ValueError('`fn` should return a Tensor or a list of Tensors.')


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
      tensor: A tensor of any type.

    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def check_min_image_dim(min_dim, image_tensor):
    """Checks that the image width/height are greater than some number.

    This function is used to check that the width and height of an image are above
    a certain value. If the image shape is static, this function will perform the
    check at graph construction time. Otherwise, if the image shape varies, an
    Assertion control dependency will be added to the graph.

    Args:
      min_dim: The minimum number of pixels along the width and height of the
               image.
      image_tensor: The image tensor to check size for.

    Returns:
      If `image_tensor` has dynamic size, return `image_tensor` with a Assert
      control dependency. Otherwise returns image_tensor.

    Raises:
      ValueError: if `image_tensor`'s' width or height is smaller than `min_dim`.
    """
    image_shape = image_tensor.get_shape()
    image_height = get_height(image_shape)
    image_width = get_width(image_shape)
    if image_height is None or image_width is None:
        shape_assert = tf.Assert(
            tf.logical_and(tf.greater_equal(tf.shape(image_tensor)[1], min_dim),
                           tf.greater_equal(tf.shape(image_tensor)[2], min_dim)),
            ['image size must be >= {} in both height and width.'.format(min_dim)])
        with tf.control_dependencies([shape_assert]):
            return tf.identity(image_tensor)

    if image_height < min_dim or image_width < min_dim:
        raise ValueError(
            'image size must be >= %d in both height and width; image dim = %d,%d' %
            (min_dim, image_height, image_width))

    return image_tensor


def get_batch_size(tensor_shape):
    """Returns batch size from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the batch size of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[0].value


def get_height(tensor_shape):
    """Returns height from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the height of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[1].value


def get_width(tensor_shape):
    """Returns width from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the width of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[2].value


def get_depth(tensor_shape):
    """Returns depth from the tensor shape.

    Args:
      tensor_shape: A rank 4 TensorShape.

    Returns:
      An integer representing the depth of the tensor.
    """
    tensor_shape.assert_has_rank(rank=4)
    return tensor_shape[3].value


def assert_shape_equal(shape_a, shape_b):
    """Asserts that shape_a and shape_b are equal.

    If the shapes are static, raises a ValueError when the shapes
    mismatch.

    If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes
    mismatch.

    Args:
      shape_a: a list containing shape of the first tensor.
      shape_b: a list containing shape of the second tensor.

    Returns:
      Either a tf.no_op() when shapes are all static and a tf.compat.v1.assert_equal() op
      when the shapes are dynamic.

    Raises:
      ValueError: When shapes are both static and unequal.
    """
    if (all(isinstance(dim, int) for dim in shape_a) and
            all(isinstance(dim, int) for dim in shape_b)):
        if shape_a != shape_b:
            raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
        else:
            return tf.no_op()
    else:
        return tf.compat.v1.assert_equal(shape_a, shape_b)


def pad_or_clip_tensor(t, length):
    """Pad or clip the input tensor along the first dimension.

    Args:
      t: the input tensor, assuming the rank is at least 1.
      length: a tensor of shape [1]  or an integer, indicating the first dimension
        of the input tensor t after processing.

    Returns:
      processed_t: the processed tensor, whose first dimension is length. If the
        length is an integer, the first dimension of the processed tensor is set
        to length statically.
    """
    return pad_or_clip_nd(t, [length] + t.shape.as_list()[1:])


def pad_or_clip_nd(tensor, output_shape):
    """Pad or Clip given tensor to the output shape.

    Args:
      tensor: Input tensor to pad or clip.
      output_shape: A list of integers / scalar tensors (or None for dynamic dim)
        representing the size to pad or clip each dimension of the input tensor.

    Returns:
      Input tensor padded and clipped to the output shape.
    """
    tensor_shape = tf.shape(tensor)
    clip_size = [
        tf.where(tensor_shape[i] - shape > 0, shape, -1)
        if shape is not None else -1 for i, shape in enumerate(output_shape)
    ]
    clipped_tensor = tf.slice(
        tensor,
        begin=tf.zeros(len(clip_size), dtype=tf.int32),
        size=clip_size)

    # Pad tensor if the shape of clipped tensor is smaller than the expected
    # shape.
    clipped_tensor_shape = tf.shape(clipped_tensor)
    trailing_paddings = [
        shape - clipped_tensor_shape[i] if shape is not None else 0
        for i, shape in enumerate(output_shape)
    ]
    paddings = tf.stack(
        [
            tf.zeros(len(trailing_paddings), dtype=tf.int32),
            trailing_paddings
        ],
        axis=1)
    padded_tensor = tf.pad(clipped_tensor, paddings=paddings)
    output_static_shape = [
        dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape
    ]
    padded_tensor.set_shape(output_static_shape)
    return padded_tensor
