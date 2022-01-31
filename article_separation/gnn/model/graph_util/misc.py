import logging
import tensorflow as tf
from tensorboard.plugins.pr_curve.summary import _create_tensor_summary
from gnn.model.graph_util import layers


def check_and_correct_interacting_nodes(interacting_nodes,
                                        edge_features,
                                        num_nodes,
                                        num_interacting_nodes,
                                        has_to_be_undirected):
    # interacting_nodes: [batch_size, max_num_interacting_nodes, 2] int
    # edge_features: [batch_size, max_num_interacting_nodes, edge_feature_dim] float
    # num_nodes: [batch_size] int
    # num_interacting_nodes: [batch_size] int
    # has_to_be_undirected: scalar bool

    batch_size = tf.shape(num_nodes)[0]

    has_edge_features = False if edge_features is None else True
    if has_edge_features:
        edge_feature_dim = edge_features.get_shape().as_list()[-1]

    if has_to_be_undirected:
        max_num_possible_interacting_nodes = 2 * tf.reduce_max(num_interacting_nodes)
    else:
        max_num_possible_interacting_nodes = tf.reduce_max(num_interacting_nodes)

    def check_and_correct_interacting_nodes_single_example(x):
        interacting_nodes_single = x[0]
        num_nodes_single = x[1]
        num_interacting_nodes_single = x[2]
        # interacting_nodes_single: [max_num_interacting_nodes, 2] int
        # num_nodes_single: scalar int
        # num_interacting_nodes_single: scalar int

        interacting_nodes_single = tf.slice(interacting_nodes_single, [0, 0], [num_interacting_nodes_single, 2])
        # interacting_nodes_single: [num_interacting_nodes_single, 2] int

        if has_edge_features:
            edge_features_single = x[3]
            # edge_features_single: [max_num_interacting_nodes, edge_feature_dim] float
            edge_features_single = tf.slice(edge_features_single, [0, 0],
                                            [num_interacting_nodes_single, edge_feature_dim])
            # edge_feature_single: [num_interacting_nodes_single, edge_feature_dim] float

        if has_to_be_undirected:
            # for undirected graph, ensure that for each edge (n_1, n_2) there exists also (n_2, n_1)
            # interacting_nodes_single_example: [num_interacting_nodes_single, 2] int
            reversed_interacting_nodes_single = tf.reverse(interacting_nodes_single, axis=[-1])
            # reversed_interacting_nodes_single: [num_interacting_nodes_single, 2] int

            interacting_nodes_single = tf.concat([interacting_nodes_single, reversed_interacting_nodes_single], axis=0)
            # interacting_nodes_single_example: [2 * num_interacting_nodes_single_example, 2] int

            if has_edge_features:
                edge_features_single = tf.tile(edge_features_single, [2, 1])
                # edge_features_single_example: [2 * num_interacting_nodes, edge_feature_dim] float

        encoded_interacting_nodes_single_full = encode_relations(interacting_nodes_single, num_nodes_single, 2)
        # encoded_interacting_nodes_single: [(2 *) num_interacting_nodes_single] int

        # remove duplicates
        encoded_interacting_nodes_single, _ = tf.unique(encoded_interacting_nodes_single_full)
        # encoded_interacting_nodes_single: [unique_num_interacting_nodes_single] int

        # remove self-loops
        self_loop_interacting_nodes = tf.range(num_nodes_single)
        self_loop_interacting_nodes = self_loop_interacting_nodes * num_nodes_single + self_loop_interacting_nodes
        no_loop_interacting_nodes = tf.sets.difference(tf.expand_dims(encoded_interacting_nodes_single, axis=0),
                                                       tf.expand_dims(self_loop_interacting_nodes, axis=0))
        encoded_interacting_nodes_single = no_loop_interacting_nodes.values

        # encoded_interacting_nodes_single: [unique_no_loops_num_interacting_nodes_single] int

        def get_index_single_interaction(encoded_interaction):
            # encoded_interaction: scalar int
            # encoded_interacting_nodes_single: [unique_no_loops_num_interacting_nodes_single] int
            # get first index where this specific interactions appears
            index = tf.where(tf.equal(encoded_interacting_nodes_single_full, encoded_interaction))
            index = tf.reshape(tf.cast(index, dtype=tf.int32), [-1])[0]
            # index: scalar int
            return index

        # for each unique interaction get the first occurence
        indices = tf.map_fn(get_index_single_interaction,
                            encoded_interacting_nodes_single,
                            dtype=tf.int32)
        # indices: [unique_no_loops_num_interacting_nodes_single] int

        corrected_interacting_nodes_single = decode_relations(encoded_interacting_nodes_single, num_nodes_single, 2)
        # corrected_interacting_nodes_single: [unique_no_loops_num_interacting_nodes_single, 2] int
        corrected_num_interacting_nodes_single = tf.shape(corrected_interacting_nodes_single)[0]

        # pad with 0
        # corrected_interacting_nodes_single: [unique_no_loops_num_interacting_nodes_single, 2] int
        pad_size = max_num_possible_interacting_nodes - corrected_num_interacting_nodes_single
        pad = tf.zeros([pad_size, 2], dtype=tf.int32)
        corrected_interacting_nodes_single = tf.concat([corrected_interacting_nodes_single, pad], axis=0)
        # corrected_interacting_nodes_single: [max_num_possible_interacting_nodes, 2] int

        if has_edge_features:
            # edge_features_single_example: [(2 *) num_interacting_nodes, edge_feature_dim] float
            corrected_edge_features_single = tf.gather(edge_features_single, indices)
            # corrected_edge_features_single: [unique_no_loops_num_interacting_nodes_single, edge_feature_dim] float
            # pad with 0
            pad = tf.zeros([pad_size, edge_feature_dim], dtype=tf.float32)
            corrected_edge_features_single = tf.concat([corrected_edge_features_single, pad], axis=0)
            # corrected_edge_features_single: [max_num_possible_interacting_nodes, edge_feature_dim] float
            return corrected_interacting_nodes_single, \
                   corrected_edge_features_single, \
                   corrected_num_interacting_nodes_single
        else:
            return corrected_interacting_nodes_single, \
                   corrected_num_interacting_nodes_single

    # interacting_nodes: [batch_size, max_num_interacting_nodes, 2] int
    # num_nodes: [batch_size] int
    # num_interacting_nodes: [batch_size] int
    # edge_features: [batch_size, max_num_interacting_nodes, edge_feature_dim] float
    if has_edge_features:
        corrected_interacting_nodes, \
        corrected_edge_features, \
        corrected_num_interacting_nodes = tf.map_fn(check_and_correct_interacting_nodes_single_example,
                                                    (
                                                        interacting_nodes, num_nodes, num_interacting_nodes,
                                                        edge_features),
                                                    dtype=(tf.int32, tf.float32, tf.int32))
    else:
        corrected_interacting_nodes, \
        corrected_num_interacting_nodes = tf.map_fn(check_and_correct_interacting_nodes_single_example,
                                                    (interacting_nodes, num_nodes, num_interacting_nodes),
                                                    dtype=(tf.int32, tf.int32))
        corrected_edge_features = None

    # corrected_num_interacting_nodes: [batch_size] int
    corrected_max_num_interacting_nodes = tf.reduce_max(corrected_num_interacting_nodes)

    if has_edge_features:
        # corrected_edge_features: [batch_size, max_num_possible_interacting_nodes, edge_feature_dim] float
        corrected_edge_features = tf.slice(corrected_edge_features, [0, 0, 0],
                                           [batch_size, corrected_max_num_interacting_nodes, edge_feature_dim])
        # corrected_edge_features: [batch_size, corrected_max_num_interacting_nodes, edge_feature_dim] float

    # corrected_interacting_nodes: [batch_size, max_num_possible_interacting_nodes, 2] int
    corrected_interacting_nodes = tf.slice(corrected_interacting_nodes, [0, 0, 0],
                                           [batch_size, corrected_max_num_interacting_nodes, 2])
    # corrected_interacting_nodes: [batch_size, corrected_max_num_interacting_nodes, 2] int
    # corrected_edge_features: [batch_size, corrected_max_num_interacting_nodes, edge_feature_dim] float
    # corrected_num_interacting_nodes: [batch_size] int
    return corrected_interacting_nodes, corrected_edge_features, corrected_num_interacting_nodes


def check_and_correct_gt_relations(gt_relations, num_nodes, gt_num_relations, num_relation_components):
    # gt_relations: [batch_size, max_num_gt_relations, num_relation_components] int
    # num_nodes: [batch_size] int
    # gt_num_relations: [batch_size] int
    # num_relation_components: scalar int

    batch_size = tf.shape(num_nodes)[0]

    max_num_possible_gt_relations = tf.reduce_max(gt_num_relations)

    def check_and_correct_gt_relations_single_example(x):
        gt_relations_single = x[0]
        num_nodes_single = x[1]
        gt_num_relations_single = x[2]
        # gt_relations_single: [max_num_gt_relations, num_relation_components] int
        # num_nodes_single: scalar int
        # gt_num_relations_single: scalar int

        gt_relations_single = tf.slice(gt_relations_single, [0, 0],
                                       [gt_num_relations_single, num_relation_components])
        # gt_relations_single: [gt_num_relations_single, num_relation_components] int

        encoded_gt_relations_single = encode_relations(gt_relations_single, num_nodes_single, num_relation_components)
        # encoded_gt_relations_single: [gt_num_relations_single] int

        # remove duplicates
        encoded_gt_relations_single, _ = tf.unique(encoded_gt_relations_single)
        # encoded_gt_relations_single: [unique_gt_num_relations_single] int

        corrected_gt_relations_single = decode_relations(encoded_gt_relations_single, num_nodes_single,
                                                         num_relation_components)
        # corrected_gt_relations_single: [unique_num_interacting_nodes_single, num_relation_components] int
        corrected_gt_num_relations_single = tf.shape(corrected_gt_relations_single)[0]

        # pad with 0
        # corrected_interacting_nodes_single_example: [unique_num_interacting_nodes_single, 2] int
        pad_size = max_num_possible_gt_relations - corrected_gt_num_relations_single
        pad = tf.zeros([pad_size, num_relation_components], dtype=tf.int32)
        corrected_gt_relations_single = tf.concat([corrected_gt_relations_single, pad], axis=0)
        # corrected_gt_relations_single: [max_num_possible_gt_relations, num_relation_components] int
        return corrected_gt_relations_single, corrected_gt_num_relations_single

    # gt_relations: [batch_size, max_num_gt_relations, num_relation_components] int
    # num_nodes: [batch_size] int
    # gt_num_relations: [batch_size] int
    corrected_gt_relations, corrected_gt_num_relations = tf.map_fn(check_and_correct_gt_relations_single_example,
                                                                   (gt_relations, num_nodes, gt_num_relations),
                                                                   dtype=(tf.int32, tf.int32))
    # corrected_gt_relations: [batch_size, max_num_possible_gt_relations, num_relation_components] int
    # corrected_gt_num_relations: [batch_size] int
    corrected_max_num_gt_relations = tf.reduce_max(corrected_gt_num_relations)
    corrected_gt_relations = tf.slice(corrected_gt_relations, [0, 0, 0],
                                      [batch_size, corrected_max_num_gt_relations, num_relation_components])
    # corrected_gt_relations: [batch_size, corrected_max_num_gt_relations, num_relation_components] int
    # corrected_gt_num_relations: [batch_size] int
    return corrected_gt_relations, corrected_gt_num_relations


def decode_relations(encoded_relations, num_nodes, num_relation_components):
    # encoded_relations: [num_relations] int
    # num_nodes: scalar int
    # num_relation_components: scalar int
    num_relations = tf.shape(encoded_relations)[0]

    mod = tf.zeros([num_relations, 0], dtype=tf.int32)
    # mod: [num_relations, 0] int
    for _ in range(num_relation_components):
        mod = tf.concat([mod, tf.expand_dims(tf.math.floormod(encoded_relations, num_nodes), axis=-1)], axis=1)
        # mod: [num_relations, x+1] int
        encoded_relations = tf.math.floordiv(encoded_relations, num_nodes)
        # encoded_relations: [num_relations] int

    # mod: [num_relations, num_relations_components] int
    relations = tf.reverse(mod, axis=[-1])
    # relations: [num_relations, num_relation_components] int

    return relations  # [num_relations, num_relation_components] int


def encode_relations(relations, num_nodes, num_relation_components):
    # relations: [num_relations, num_relation_components] int
    # num_nodes: scalar int
    # num_relation_components: scalar int
    num_relations = tf.shape(relations)[0]

    relation_components = list(reversed(tf.unstack(relations, axis=-1)))
    # relation_components: list of [num_relations] int and length num_relation_components

    encoded_relations = tf.zeros([num_relations], dtype=tf.int32)
    for comp in range(num_relation_components):
        encoded_relations += relation_components[comp] * (num_nodes ** comp)

    return encoded_relations  # [num_relations] int


def normalize_visual_regions(visual_regions, true_image_shape, pad_image_height, pad_image_width):
    # visual_regions: [batch_size, max_num_nodes, 2, max_numpoints_region] float
    # true_image_shape: [batch_size, 3] int
    # pad_image_height, pad_image_width: int

    batch_size = tf.shape(visual_regions)[0]
    max_num_nodes = tf.shape(visual_regions)[1]
    max_num_points_region = tf.shape(visual_regions)[3]

    pad_image_shape = tf.tile(tf.expand_dims(tf.stack([pad_image_height, pad_image_width]), axis=0), [batch_size, 1])
    # pad_image_shape: [batch_size, 2] int
    scale = tf.cast(tf.truediv(tf.slice(true_image_shape, [0, 0], [batch_size, 2]), pad_image_shape), dtype=tf.float32)
    scale = tf.reverse(scale, axis=[-1])
    # scale: [batch_size, 2] float
    scale = tf.tile(tf.expand_dims(tf.expand_dims(scale, axis=1), axis=-1),
                    [1, max_num_nodes, 1, max_num_points_region])
    # scale: [batch_size, max_num_nodes, 2, max_numpoints_region] float

    normalized_visual_regions = tf.multiply(visual_regions, scale)
    # normalized visual_regions: [batch_size, max_num_nodes, 2, max_numpoints_region] float
    return normalized_visual_regions


def normalize_image(image, img_shape):
    # dynamic shape values (calculated during runtime)
    shape_dyn_stat = combined_static_and_dynamic_shape(image)
    # image normalization
    image_crop = tf.image.crop_to_bounding_box(image, 0, 0, img_shape[0], img_shape[1])
    image_norm = layers.per_image_standardization(image_crop)
    image_pad = tf.image.pad_to_bounding_box(image_norm, 0, 0, shape_dyn_stat[0], shape_dyn_stat[1])
    return image_pad


def assign_visual_node_features(feature_maps, regions, num_points, layer_compressed_dims, is_training,
                                dropout_feature_map=0.0, dropout_visual_feature_compression=0.0):
    # feature_maps: list of [batch_size, feature_map_i_max_height, feature_map_i_max_width, feature_map_i_channels] float
    # regions: [batch_size, max_num_nodes, 2, max_numpoints_region] float
    # num_points: [batch_size, max_num_nodes] int

    # reduce complex regions to paraxial bounding rectangle
    batch_size = tf.shape(regions)[0]
    max_num_nodes = tf.shape(regions)[1]
    max_numpoints_region = tf.shape(regions)[3]
    regions = tf.reshape(regions, [-1, 2, max_numpoints_region])
    # regions: [batch_size * max_num_nodes, 2, max_num_points_region] float
    num_points = tf.reshape(num_points, [-1])
    # num_points: [batch_size * max_num_nodes] int

    bounds = tf.map_fn(lambda x: make_paraxial_rectangular(x[0], x[1]), elems=(regions, num_points), dtype=tf.float32)
    # bounds: [batch_size * max_num_nodes, 4] float

    xmin, xmax, ymin, ymax = tf.unstack(bounds, axis=1)
    # xmin, xmax, ymin, ymax: [batch_size * max_num_nodes] float

    visual_feature_list = []
    visual_feature_dim_list = []
    for fm_idx, feature_map in enumerate(feature_maps):
        # feature_map: [batch_size, feature_map_i_max_height, feature_map_i_max_width, feature_map_i_channels] float
        if dropout_feature_map > 0.0:
            feature_map = layers.dropout(feature_map, is_training, dropout_feature_map)

        shape_dyn_stat = combined_static_and_dynamic_shape(feature_map)

        fm_height = shape_dyn_stat[1]
        fm_width = shape_dyn_stat[2]
        fm_channels = shape_dyn_stat[3]

        # fm_height = tf.Print(fm_height, [fm_height, fm_width, fm_channels],
        #                      message="fm_shape: ", summarize=1024)

        # fm_height = tf.Print(fm_height, [xmin, xmax, ymin, ymax],
        #                      message="xy: ", summarize=1024)

        fm_xmin = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(xmin, tf.cast(fm_width, dtype=tf.float32))), dtype=tf.int32),
                       fm_width - 1), tf.constant(0, dtype=tf.int32))
        fm_xmax = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(xmax, tf.cast(fm_width, dtype=tf.float32))), dtype=tf.int32),
                       fm_width - 1), tf.constant(0, dtype=tf.int32))
        fm_ymin = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(ymin, tf.cast(fm_height, dtype=tf.float32))), dtype=tf.int32),
                       fm_height - 1), tf.constant(0, dtype=tf.int32))
        fm_ymax = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(ymax, tf.cast(fm_height, dtype=tf.float32))), dtype=tf.int32),
                       fm_height - 1), tf.constant(0, dtype=tf.int32))
        # fm_xmin, fm_xmax, fm_ymin, fm_ymax: [batch_size * max_num_nodes] int

        fm_num_x = tf.maximum(fm_xmax - fm_xmin + 1, tf.constant(1, dtype=tf.int32))
        fm_num_y = tf.maximum(fm_ymax - fm_ymin + 1, tf.constant(1, dtype=tf.int32))
        # fm_num_x, fm_num_y: [batch_size * max_num_nodes] int

        batch_range = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, limit=batch_size), axis=-1), [1, max_num_nodes]),
                                 [-1])

        # batch_range: [batch_size * max_num_nodes] int

        def visual_feature_extraction(i_c, visual_feature_c):
            # inst, node_fm_xmin, node_fm_ymin, node_fm_num_x, node_fm_num_y: int
            return [i_c + 1, tf.concat([visual_feature_c, tf.expand_dims(tf.reduce_max(
                tf.slice(feature_map, [batch_range[i_c], fm_ymin[i_c], fm_xmin[i_c], 0],
                         [1, fm_num_y[i_c], fm_num_x[i_c], fm_channels]), axis=[0, 1, 2]), axis=0)], axis=0)]

        visual_feature = tf.zeros([0, fm_channels], dtype=tf.float32)
        i = tf.constant(0, dtype=tf.int32)
        _, visual_feature = tf.while_loop(
            lambda i, vf: tf.less(i, batch_size * max_num_nodes),
            visual_feature_extraction,
            loop_vars=[i, visual_feature],
            shape_invariants=[i.get_shape(), tf.TensorShape([None, fm_channels])],
            swap_memory=True
        )

        # visual_feature = tf.Print(visual_feature, [tf.shape(visual_feature)], message="result: ", summarize=1024)

        compressed_fm_channels = layer_compressed_dims[fm_idx]

        with tf.compat.v1.variable_scope('visual_node_feature_compression_fm_' + str(fm_idx)):
            # visual_feature: [batch_size * max_num_nodes, feature_map_i_channels] float
            visual_feature = layers.ff_layer(visual_feature, outD=compressed_fm_channels, activation=layers.relu,
                                             use_bias=True, is_training=is_training, reuse=None)
            # visual_feature: [batch_size * max_num_nodes, compressed_feature_map_i_channels] float

        visual_feature = tf.reshape(visual_feature, [batch_size, max_num_nodes, compressed_fm_channels])

        if dropout_visual_feature_compression > 0.0:
            visual_feature = layers.dropout(visual_feature, is_training, dropout_visual_feature_compression)

        # visual_feature: [batch_size, max_num_nodes, compressed_feature_map_i_channels] float
        visual_feature_list.append(visual_feature)
        visual_feature_dim_list.append(compressed_fm_channels)

    # visual_feature_list: list of [batch_size, max_num_nodes, feature_map_i_channels] float
    return visual_feature_list, visual_feature_dim_list


def assign_visual_edge_features(feature_maps, regions, num_points, layer_compressed_dims, is_training,
                                dropout_feature_map=0.0, dropout_visual_feature_compression=0.0):
    # feature_maps: list of [batch_size, feature_map_i_max_height, feature_map_i_max_width, feature_map_i_channels] float
    # regions: [batch_size, max_num_interacting_nodes, 2, max_numpoints_region] float
    # num_points: [batch_size, max_num_interacting_nodes] int

    # reduce complex regions to paraxial bounding rectangle
    batch_size = tf.shape(regions)[0]
    max_num_nodes = tf.shape(regions)[1]
    max_numpoints_region = tf.shape(regions)[3]
    regions = tf.reshape(regions, [-1, 2, max_numpoints_region])
    # regions: [batch_size * max_num_nodes, 2, max_num_points_region] float
    num_points = tf.reshape(num_points, [-1])
    # num_points: [batch_size * max_num_nodes] int

    bounds = tf.map_fn(lambda x: make_paraxial_rectangular(x[0], x[1]), elems=(regions, num_points), dtype=tf.float32)
    # bounds: [batch_size * max_num_nodes, 4] float

    xmin, xmax, ymin, ymax = tf.unstack(bounds, axis=1)
    # xmin, xmax, ymin, ymax: [batch_size * max_num_nodes] float

    visual_feature_list = []
    visual_feature_dim_list = []
    for fm_idx, feature_map in enumerate(feature_maps):
        # feature_map: [batch_size, feature_map_i_max_height, feature_map_i_max_width, feature_map_i_channels] float
        if dropout_feature_map > 0.0:
            feature_map = layers.dropout(feature_map, is_training, dropout_feature_map)

        shape_dyn_stat = combined_static_and_dynamic_shape(feature_map)

        fm_height = shape_dyn_stat[1]
        fm_width = shape_dyn_stat[2]
        fm_channels = shape_dyn_stat[3]

        # fm_height = tf.Print(fm_height, [fm_height, fm_width, fm_channels],
        #                      message="fm_shape: ", summarize=1024)

        # fm_height = tf.Print(fm_height, [xmin, xmax, ymin, ymax],
        #                      message="xy: ", summarize=1024)

        fm_xmin = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(xmin, tf.cast(fm_width, dtype=tf.float32))), dtype=tf.int32),
                       fm_width - 1), tf.constant(0, dtype=tf.int32))
        fm_xmax = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(xmax, tf.cast(fm_width, dtype=tf.float32))), dtype=tf.int32),
                       fm_width - 1), tf.constant(0, dtype=tf.int32))
        fm_ymin = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(ymin, tf.cast(fm_height, dtype=tf.float32))), dtype=tf.int32),
                       fm_height - 1), tf.constant(0, dtype=tf.int32))
        fm_ymax = tf.maximum(
            tf.minimum(tf.cast(tf.floor(tf.multiply(ymax, tf.cast(fm_height, dtype=tf.float32))), dtype=tf.int32),
                       fm_height - 1), tf.constant(0, dtype=tf.int32))
        # fm_xmin, fm_xmax, fm_ymin, fm_ymax: [batch_size * max_num_nodes] int

        fm_num_x = tf.maximum(fm_xmax - fm_xmin + 1, tf.constant(1, dtype=tf.int32))
        fm_num_y = tf.maximum(fm_ymax - fm_ymin + 1, tf.constant(1, dtype=tf.int32))
        # fm_num_x, fm_num_y: [batch_size * max_num_nodes] int

        batch_range = tf.reshape(tf.tile(tf.expand_dims(tf.range(0, limit=batch_size), axis=-1), [1, max_num_nodes]),
                                 [-1])

        # batch_range: [batch_size * max_num_nodes] int

        def visual_feature_extraction(i_c, visual_feature_c):
            # inst, node_fm_xmin, node_fm_ymin, node_fm_num_x, node_fm_num_y: int
            return [i_c + 1, tf.concat([visual_feature_c, tf.expand_dims(tf.reduce_max(
                tf.slice(feature_map, [batch_range[i_c], fm_ymin[i_c], fm_xmin[i_c], 0],
                         [1, fm_num_y[i_c], fm_num_x[i_c], fm_channels]), axis=[0, 1, 2]), axis=0)], axis=0)]

        visual_feature = tf.zeros([0, fm_channels], dtype=tf.float32)
        i = tf.constant(0, dtype=tf.int32)
        _, visual_feature = tf.while_loop(
            lambda i, vf: tf.less(i, batch_size * max_num_nodes),
            visual_feature_extraction,
            loop_vars=[i, visual_feature],
            shape_invariants=[i.get_shape(), tf.TensorShape([None, fm_channels])],
            swap_memory=True
        )

        # visual_feature = tf.Print(visual_feature, [tf.shape(visual_feature)], message="result: ", summarize=1024)

        compressed_fm_channels = layer_compressed_dims[fm_idx]

        with tf.compat.v1.variable_scope('visual_edge_feature_compression_fm_' + str(fm_idx)):
            # visual_feature: [batch_size * max_num_nodes, feature_map_i_channels] float
            visual_feature = layers.ff_layer(visual_feature, outD=compressed_fm_channels, activation=layers.relu,
                                             use_bias=True, is_training=is_training, reuse=None)
            # visual_feature: [batch_size * max_num_nodes, compressed_feature_map_i_channels] float

        visual_feature = tf.reshape(visual_feature, [batch_size, max_num_nodes, compressed_fm_channels])

        if dropout_visual_feature_compression > 0.0:
            visual_feature = layers.dropout(visual_feature, is_training, dropout_visual_feature_compression)

        # visual_feature: [batch_size, max_num_nodes, compressed_feature_map_i_channels] float
        visual_feature_list.append(visual_feature)
        visual_feature_dim_list.append(compressed_fm_channels)

    # visual_feature_list: list of [batch_size, max_num_nodes, feature_map_i_channels] float
    return visual_feature_list, visual_feature_dim_list


def make_paraxial_rectangular_tmp(region, num_points):
    region = tf.slice(region, [0, 0], [2, num_points])
    # region: [2, num_points] float
    xymin = tf.reduce_min(region, axis=-1)
    xymax = tf.reduce_max(region, axis=-1)

    xmin = xymin[0]
    ymin = xymin[1]
    xmax = xymax[0]
    ymax = xymax[1]

    # xmin = tf.Print(xmin, [xmin, xmax, ymin, ymax, xymin, xymax, region],
    #                 message="reg: ", summarize=1024)

    return tf.stack([xmin, xmax, ymin, ymax])


def make_paraxial_rectangular(region, num_points):
    # region: [2, max_numpoints_region] float
    # num_points: int
    return (tf.cond(tf.equal(num_points, tf.constant(0, dtype=tf.int32)),
                    lambda: tf.zeros([4], dtype=tf.float32),
                    lambda: make_paraxial_rectangular_tmp(region, num_points)))


def drop_edge(interacting_nodes, is_training, rate=None, keep_prob=None, name='drop_edge'):
    # interacting_nodes: [batch_size, max_num_interacting_nodes, 2] int
    if rate and keep_prob:
        logging.error("Use either keep_prob or rate for drop_edge!")
        exit(1)
    if keep_prob:
        rate = 1.0 - keep_prob
    # rate is set to zero when not in training
    rate = tf.cast(is_training, dtype=tf.float32) * rate
    # noise_shape drops entire edges instead of single elements
    noise_shape = [tf.shape(interacting_nodes)[0], tf.shape(interacting_nodes)[1], 1]
    dropped = tf.cast(tf.nn.dropout(
        tf.cast(interacting_nodes, tf.float32), rate=rate, noise_shape=noise_shape, name=name) * (1 - rate), tf.int32)
    return dropped


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


def curve_streaming_op(name, labels, predictions, num_thresholds=201, weights=None, metrics_collections=None,
                       updates_collections=None, display_name=None, description=None, curve='PR'):
    """Computes a true_positive_rate-false_positive_rate curve summary across batches of data.
    for ROC-Curve seee https://towardsdatascience.com/understanding-auc-roc-curve-68b2303cc9c5
    This function is similar to op() above, but can be used to compute the PR
    curve across multiple batches of labels and predictions, in the same style
    as the metrics found in tf.metrics.

    This function creates multiple local variables for storing true positives,
    true negative, etc. accumulated over each batch of data, and uses these local
    variables for computing the final PR curve summary. These variables can be
    updated with the returned update_op.

    Args:
      name: A tag attached to the summary. Used by TensorBoard for organization.
      labels: The ground truth values, a `Tensor` whose dimensions must match
        `predictions`. Will be cast to `bool`.
      predictions: A floating point `Tensor` of arbitrary shape and whose values
        are in the range `[0, 1]`.
      num_thresholds: The number of evenly spaced thresholds to generate for
        computing the PR curve. Defaults to 201.
      weights: Optional `Tensor` whose rank is either 0, or the same rank as
        `labels`, and must be broadcastable to `labels` (i.e., all dimensions must
        be either `1`, or the same as the corresponding `labels` dimension).
      metrics_collections: An optional list of collections that `auc` should be
        added to.
      updates_collections: An optional list of collections that `update_op` should
        be added to.
      display_name: Optional name for this summary in TensorBoard, as a
          constant `str`. Defaults to `name`.
      description: Optional long-form description for this summary, as a
          constant `str`. Markdown is supported. Defaults to empty.
      curve: can be 'PR' or 'ROC' or 'BOTH'

    Returns:
      curve: for curve in ['PR','ROC']: A string `Tensor` containing a single value: the
        serialized curve Tensor summary. The summary contains a
        float32 `Tensor` of dimension (6, num_thresholds). The first
        dimension (of length 6) is of the order: true positives, false
        positives, true negatives, false negatives, y-axis, x-axis.
      update_op: for curve in ['PR','ROC'] An operation that updates the summary with the latest data.
    """
    _MINIMUM_COUNT = 1e-7
    thresholds = [i / float(num_thresholds - 1) for i in range(num_thresholds)]

    with tf.name_scope(name, values=[labels, predictions, weights]):
        tp, update_tp = tf.compat.v1.metrics.true_positives_at_thresholds(labels=labels, predictions=predictions,
                                                                          thresholds=thresholds, weights=weights, )
        fp, update_fp = tf.compat.v1.metrics.false_positives_at_thresholds(labels=labels, predictions=predictions,
                                                                           thresholds=thresholds, weights=weights, )
        tn, update_tn = tf.compat.v1.metrics.true_negatives_at_thresholds(labels=labels, predictions=predictions,
                                                                          thresholds=thresholds, weights=weights, )
        fn, update_fn = tf.compat.v1.metrics.false_negatives_at_thresholds(labels=labels, predictions=predictions,
                                                                           thresholds=thresholds, weights=weights, )

        def compute_summary_precision_recall(tp, fp, tn, fn, collections):
            precision = tp / tf.maximum(_MINIMUM_COUNT, tp + fp)
            recall = tp / tf.maximum(_MINIMUM_COUNT, tp + fn)

            return _create_tensor_summary(name, tp, fp, tn, fn, precision, recall, num_thresholds, display_name,
                                          description, collections, )

        def compute_summary_roc(tp, fp, tn, fn, collections):
            true_positive_rate = tp / tf.maximum(_MINIMUM_COUNT, tp + fn)
            false_positive_rate = fp / tf.maximum(_MINIMUM_COUNT, fp + tn)

            return _create_tensor_summary(name, tp, fp, tn, fn, true_positive_rate, false_positive_rate, num_thresholds,
                                          display_name, description, collections, )

        update_op = tf.group(update_tp, update_fp, update_tn, update_fn)
        if updates_collections:
            for collection in updates_collections:
                tf.compat.v1.add_to_collection(collection, update_op)

        if curve == 'PR':
            with tf.name_scope("PR", values=[labels, predictions, weights]):
                pr_curve = compute_summary_precision_recall(tp, fp, tn, fn, metrics_collections)
            return pr_curve, update_op
        if curve == 'ROC':
            with tf.name_scope("ROC", values=[labels, predictions, weights]):
                roc_curve = compute_summary_roc(tp, fp, tn, fn, metrics_collections)
            return roc_curve, update_op

        with tf.name_scope("PR", values=[labels, predictions, weights]):
            pr_curve = compute_summary_precision_recall(tp, fp, tn, fn, metrics_collections)
        with tf.name_scope("ROC", values=[labels, predictions, weights]):
            roc_curve = compute_summary_roc(tp, fp, tn, fn, metrics_collections)

        return (pr_curve, update_op), (roc_curve, update_op)
#
#
# if __name__ == '__main__':
#     # inter_nodes = tf.constant([[[0, 0], [0, 1], [0, 2], [1, 1], [2, 0], [0, 0]],
#     #                            [[0, 0], [0, 1], [1, 0], [1, 1], [2, 1], [2, 0]]])
#     # n_nodes = tf.constant([3, 4])
#     # num_int = tf.constant([5, 6])
#     # int_nodes = tf.constant([[[0, 0], [0, 1], [0, 2], [1, 1], [2, 0], [2, 2], [1, 3], [3, 3], [0, 0]]])
#     # num_nodes = tf.constant([4])
#     # num_int = tf.constant([8])
#
#     # cor_int_nodes, _, cor_num_int = check_and_correct_interacting_nodes(interacting_nodes=inter_nodes,
#     #                                                                     edge_features=None,
#     #                                                                     num_nodes=n_nodes,
#     #                                                                     num_interacting_nodes=num_int,
#     #                                                                     has_to_be_undirected=True)
#
#     sess = tf.Session()
#     # nodes, cor_nodes, num, cor_num = sess.run([inter_nodes, cor_int_nodes, num_int, cor_num_int])
#     #
#     # print("number_interacting_nodes ", num)
#     # print("number_corrected_interacting_nodes ", cor_num)
#     # print("nodes ", nodes)
#     # print("corrected_nodes ", cor_nodes)
#
#     # [max_num_edges, 2, max_numpoints_region]
#     visual_region_1 = tf.constant([[[0.2, 0.2, 0.4, 0.4], [0.2, 0.6, 0.2, 0.6]]])
#     visual_region_2 = tf.constant([[[0.5, 0.5, 0.8, 0.8], [0.4, 0.3, 0.4, 0.3]]])
#     # [batch_size, max_num_edges, 2, max_numpoints_region]
#     visual_regions = tf.stack([visual_region_1, visual_region_2], axis=0)
#
#     # true_image_shape: [batch_size, 3] int
#     true_shape = tf.constant([[2000, 1200, 1], [1000, 800, 1]])
#     # pad_image_height, pad_image_width: int
#     pad_h = 2000
#     pad_w = 1200
#
#     normal_regions = normalize_visual_regions(visual_regions, true_shape, pad_h, pad_w)
#
#     # list of[batch_size, feature_map_i_max_height, feature_map_i_max_width, feature_map_i_channels]
#     feat_1 = tf.tile(tf.reshape(tf.range(1, 4, dtype=tf.float32), shape=[1, 1, -1]), multiples=[10, 5, 1])
#     feat_2 = tf.tile(tf.reshape(tf.range(1, 4, dtype=tf.float32) * 10, shape=[1, 1, -1]), multiples=[10, 5, 1])
#     feats = tf.stack([feat_1, feat_2], axis=0)
#
#     num_points = tf.constant([[4], [4]])
#     compress_dims = [3]
#     visual, visual_dim = assign_visual_node_features([feats], normal_regions, num_points, compress_dims, False)
#
#     sess.run(tf.compat.v1.global_variables_initializer())
#     r, f, v = sess.run([visual_regions, feats, visual])
#     print(r.shape)
#     print(r)
#     print(f.shape)
#     # print(f)
#     print(v[0].shape)
#     print(v)
#
#     # tmp1 = tf.constant(1, shape=[2, 10, 5])
#     # tmp2 = tf.constant(2, shape=[2, 10, 5])
#     # tmp3 = tf.constant(3, shape=[2, 10, 5])
#     # tmp = tf.stack([tmp1, tmp2, tmp3], axis=-1)
#     # inds = [[0, 0, 0], [0, 0, 1]]
#     # gath = tf.gather_nd(tmp, inds)
#     # g = sess.run(gath)
#     # print(g.shape)
#     # print(g)
