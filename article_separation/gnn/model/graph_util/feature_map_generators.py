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

"""Functions to generate a list of feature maps based on image features.

Provides several feature map generators that can be used to build object
detection feature extractors.

Object detection feature extractors usually are built by stacking two components
- A base feature extractor such as Inception V3 and a feature map generator.
Feature map generators build on the base feature extractors and produce a list
of final feature maps.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import collections
import tensorflow as tf
from gnn.model.graph_util import layers


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


def get_depth_fn(depth_multiplier, min_depth):
    """Builds a callable to compute depth (output channels) of conv filters.

    Args:
      depth_multiplier: a multiplier for the nominal depth.
      min_depth: a lower bound on the depth of filters.

    Returns:
      A callable that takes in a nominal depth and returns the depth to use.
    """

    def multiply_depth(depth):
        new_depth = int(depth * depth_multiplier)
        return max(new_depth, min_depth)

    return multiply_depth


def multi_resolution_feature_maps(feature_map_layout, is_training,
                                  insert_1x1_conv, image_features,
                                  pool_residual=False):
    """Generates multi resolution feature maps from input image/backbone features.

    Generates multi-scale feature maps for detection as in the SSD papers by
    Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1. (we adapted it a little bit [Tobi])

    More specifically, it performs the following three tasks:
    1) If a layer name is provided in the configuration and its corresponding depth is -1, returns that (backbone) layer as a
       feature map.
    2) If a layer name is provided in the configuration and its corresponding depth is NOT -1, the (backbone) layer is convolved
       and returned as feature map.
    3) If a layer name is left as an empty string, constructs a new feature map
       based on the last calculated feature map and the provided depth. Of Note: A subsampling of 2x2 is performed using strided conv.

    An example of the configuration for Inception V3:

    {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, 128, -1, 512, 256, 128]
    }

    Parametrization (in start script) should look like: --feature_map_generation_params
    from_layer=[Mixed_5d,Mixed_6e,Mixed_7c,,,] layer_depth=[-1,128,-1,512,256,128]
    NO SPACES, NO ', NO "

    Args:
      feature_map_layout: Dictionary of specifications for the feature map
        layouts (see above)
        Additionally: Convolution kernel size is set to 3 by default, and can be
        customized by 'conv_kernel_size' parameter (similarily, 'conv_kernel_size'
        should be set to -1 if 'from_layer' is specified). The created convolution
        operation will be a normal 2D convolution by default, and a depthwise
        convolution followed by 1x1 convolution if 'use_depthwise' is set to True.
      is_training: A boolean indicating we are in training or in validation mode
      insert_1x1_conv: A boolean indicating whether an additional 1x1 convolution
        should be inserted before shrinking the feature map.
      image_features: A dictionary of handles to activation tensors from the
        base feature extractor.
      pool_residual: Whether to add an average pooling layer followed by a
        residual connection between subsequent feature maps when the channel
        depth match. For example, with option 'layer_depth': [-1, 512, 256, 256],
        a pooling and residual layer is added between the third and forth feature
        map. This option is better used with Weight Shared Convolution Box
        Predictor when all feature maps have the same channel depth to encourage
        more consistent features across multi-scale feature maps.

    Returns:
      feature_maps: an OrderedDict mapping keys (feature map names) to
        tensors where each tensor has shape [batch, height_i, width_i, depth_i].

    Raises:
      ValueError: if the number entries in 'from_layer' and
        'layer_depth' do not match.
      ValueError: if the generated layer does not have the same resolution
        as specified.
    """
    assert len(feature_map_layout['from_layer']) == len(feature_map_layout['layer_depth'])
    feature_map_keys = []
    feature_maps = []
    base_from_layer = ''
    use_explicit_padding = False
    if 'use_explicit_padding' in feature_map_layout:
        use_explicit_padding = feature_map_layout['use_explicit_padding']
    use_depthwise = False
    if 'use_depthwise' in feature_map_layout:
        use_depthwise = feature_map_layout['use_depthwise']
    for index, from_layer in enumerate(feature_map_layout['from_layer']):
        layer_depth = feature_map_layout['layer_depth'][index]
        conv_kernel_size = 3
        if 'conv_kernel_size' in feature_map_layout:
            conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
        if from_layer and layer_depth == -1:
            feature_map = image_features[from_layer]
            base_from_layer = from_layer
            feature_map_keys.append(from_layer)
        else:
            if from_layer:
                pre_layer = image_features[from_layer]
                stride = 1
            else:
                pre_layer = feature_maps[-1]
                stride = 2
            pre_layer_depth = pre_layer.get_shape().as_list()[3]
            intermediate_layer = pre_layer
            if insert_1x1_conv:
                layer_name = f'{base_from_layer}_1_Conv2d_{index}_1x1_{layer_depth / 2}'
                intermediate_layer = layers.conv2d(pre_layer, kernel_size=[1, 1], filters=(layer_depth / 2),
                                                   is_training=is_training, strides=[1, 1, 1, 1],
                                                   padding='SAME', name=layer_name)

            layer_name = f'{base_from_layer}_2_Conv2d_{index}_{conv_kernel_size}x{conv_kernel_size}_s2_{layer_depth}'
            padding = 'SAME'
            if use_explicit_padding:
                padding = 'VALID'
                intermediate_layer = fixed_padding(
                    intermediate_layer, conv_kernel_size)
            if use_depthwise:
                feature_map = layers.sep_conv2d(intermediate_layer,
                                                kernel_size=[conv_kernel_size, conv_kernel_size],
                                                filters=intermediate_layer.shape[3],
                                                depth_multiplier=1, is_training=is_training,
                                                strides=[1, stride, stride, 1],
                                                padding=padding, name=layer_name + '_depthwise')
                feature_map = layers.conv2d(feature_map,
                                            kernel_size=[1, 1],
                                            filters=layer_depth,
                                            is_training=is_training, strides=[1, 1, 1, 1],
                                            padding='SAME', name=layer_name)

                if pool_residual and pre_layer_depth == layer_depth:
                    feature_map += layers.avg_pool2d(feature_map, ksize=[1, 3, 3, 1], strides=[1, stride, stride, 1],
                                                     padding='SAME',
                                                     name=layer_name + '_pool')
            else:
                feature_map = layers.conv2d(intermediate_layer,
                                            kernel_size=[conv_kernel_size, conv_kernel_size],
                                            filters=layer_depth,
                                            is_training=is_training, strides=[1, stride, stride, 1],
                                            padding=padding, name=layer_name)
            feature_map_keys.append(layer_name)
        feature_maps.append(feature_map)
    return collections.OrderedDict(
        [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])
