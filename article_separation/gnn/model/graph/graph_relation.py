import logging
import tensorflow as tf
from gnn.model.model_base import GraphBase
from gnn.model.graph.graph_gnn import GraphGNN
from gnn.model.graph_util.layers import mlp
from gnn.model.graph_util.misc import normalize_visual_regions, normalize_image, assign_visual_edge_features, \
    assign_visual_node_features, drop_edge, combined_static_and_dynamic_shape
from backbones.backbones import Backbones
from gnn.model.graph_util import feature_map_generators
from utils.flags import update_params


class GraphRelation(GraphBase):
    def __init__(self, params):
        super(GraphRelation, self).__init__(params)
        self._params = params

        self._image_input = self._flags.image_input
        if self._image_input:
            backbone = self._flags.backbone
            # This graph gets the input image and generates (multi-purpose) feature maps
            self._graph_feat = Backbones(backbone, self._params)
            self._mvn = self._flags.mvn

            # Default configuration for the feature map generation given the backbone feature maps (this has to FIT to the backbone.)
            # See model_fn.model_fn_img.model_fn_objdet.graphs.feature.feature_map_genarators.multi_resolution_feature_maps for explanation
            self._feature_map_generation_params = dict()
            self._feature_map_generation_params["from_layer"] = ['Mixed_5d', 'Mixed_6e', 'Mixed_7c']
            self._feature_map_generation_params["layer_depth"] = [-1, -1, -1]
            self._feature_map_generation_params["layer_compressed_dim"] = [16, 16, 16]
            self._feature_map_generation_params = update_params(self._feature_map_generation_params,
                                                                self._flags.feature_map_generation_params,
                                                                "FeatureMap Generator")
            self.graph_params["dropout_feature_map"] = 0.0
            self.graph_params["dropout_visual_feature_compression"] = 0.0
            self.graph_params["assign_visual_features_to_nodes"] = True
            self.graph_params["assign_visual_features_to_edges"] = False

        self.graph_params["dropout_edges"] = 0.0
        self.graph_params["dropout_classifier"] = 0.0

        # Default configuration for the relation graph
        # Updating of the default params if provided via flags as a dict
        self.graph_params = update_params(self.graph_params, self._flags.rel_params, "RelationGraph")

        self._gnn1 = GraphGNN(params=self._params, gnn_params=self._flags.gnn_params_1,
                              message_fn_params=self._flags.message_fn_params_1,
                              update_fn_params=self._flags.update_fn_params_1)

    def print_params(self):
        sorted_dict = sorted(self.graph_params.items(), key=lambda kv: kv[0])
        if len(sorted_dict) > 0:
            logging.info("graph_params:")
            for a in sorted_dict:
                logging.info(f"  {a[0]}: {a[1]}")

        if self._image_input:
            self._graph_feat.print_params()
            sorted_dict = sorted(self._feature_map_generation_params.items(), key=lambda kv: kv[0])
            if len(sorted_dict) > 0:
                logging.info("feature_map_generator_params:")
                for a in sorted_dict:
                    logging.info(f"  {a[0]}: {a[1]}")

        self._gnn1.print_params()

    def infer(self, inputs, is_training):
        # ===== GNN
        # prepare GNN input
        gnn_inputs = dict()
        gnn_inputs['num_nodes'] = inputs['num_nodes']  # [batch_size] int
        # inputs['interacting_nodes'] : [batch_size, max_num_interacting_nodes, 2] int
        gnn_inputs['interacting_nodes'] = inputs['interacting_nodes']
        # graph_inputs['interacting_nodes']: [batch_size, max_num_interacting_nodes, 2] int
        gnn_inputs['num_interacting_nodes'] = inputs['num_interacting_nodes']
        # gnn_inputs['num_interacting_nodes']: [batch_size] int
        if 'node_features' in inputs:
            gnn_inputs['node_features'] = inputs['node_features']
            # gnn_inputs['node_features']: [batch_size, max_num_nodes, node_feature_dim] float
        if 'edge_features' in inputs:
            gnn_inputs['edge_features'] = inputs['edge_features']
            # gnn_inputs['edge_features']: [batch_size, max_num_edges, edge_feature_dim] float

        if self._image_input and 'image' in inputs and 'image_shape' in inputs:
            preprocessed_inputs = inputs['image']
            # preprocessed_inputs: [batch_size, image_max_height, image_max_width, image_num_channels] float
            image_shape = inputs['image_shape']
            # image_shape: [batch_size, 3] int

            if self._mvn:
                elems = (preprocessed_inputs, image_shape)
                preprocessed_inputs = tf.map_fn(lambda imP: normalize_image(imP[0], imP[1]), elems=elems,
                                                dtype=tf.float32)
            pad_image_shape = combined_static_and_dynamic_shape(preprocessed_inputs)

            # Updates the backbone feature extractor --> calculates a list of features maps
            _, image_features = self._graph_feat.infer(preprocessed_inputs, is_training)

            # Takes the feature maps and generates multi_resolution maps
            feature_maps = feature_map_generators.multi_resolution_feature_maps(
                feature_map_layout=self._feature_map_generation_params,
                is_training=is_training,
                insert_1x1_conv=True,
                image_features=image_features)
            feature_maps = feature_maps.values()
            # feature_maps: list of [batch_size, feature_map_i_max_height, feature_map_i_max_width, feature_map_i_channels] float

            if self.graph_params["assign_visual_features_to_nodes"]:
                visual_regions_nodes = inputs['visual_regions_nodes']
                # points relative to the original image height and width, index 0: x, index 1: y
                # visual_regions_nodes: [batch_size, max_num_nodes, 2, max_numpoints_region] float
                num_points_visual_regions_nodes = inputs['num_points_visual_regions_nodes']
                # num_points_visual_regions_nodes: [batch_size, max_num_nodes] int

                # normalize visual regions with regards to batch padding
                visual_regions_nodes = normalize_visual_regions(visual_regions_nodes, image_shape,
                                                                pad_image_shape[1], pad_image_shape[2])
                # visual_regions_nodes: [batch_size, max_num_nodes, 2, max_numpoints_region] float

                visual_node_feature_list, \
                visual_node_feature_dim_list = \
                    assign_visual_node_features(feature_maps, visual_regions_nodes, num_points_visual_regions_nodes,
                                                self._feature_map_generation_params["layer_compressed_dim"],
                                                is_training,
                                                dropout_feature_map=self.graph_params["dropout_feature_map"],
                                                dropout_visual_feature_compression=self.graph_params[
                                                    "dropout_visual_feature_compression"])
                # visual_feature_list: list of [batch_size, max_num_nodes, feature_map_i_channels] float

                visual_node_features = tf.concat(visual_node_feature_list, axis=-1)
                visual_node_feature_dim = sum(visual_node_feature_dim_list)
                # visual_node_features: [batch_size, max_num_nodes, visual_feature_dim] float

                # add the visual features to the node features
                if 'node_features' in gnn_inputs:
                    gnn_inputs['node_features'] = tf.concat([gnn_inputs['node_features'], visual_node_features],
                                                            axis=-1)
                else:
                    gnn_inputs['node_features'] = visual_node_features

            if self.graph_params["assign_visual_features_to_edges"]:
                visual_regions_edges = inputs['visual_regions_edges']
                # points relative to the original image height and width, index 0: x, index 1: y
                # visual_regions_edges: [batch_size, max_num_interacting_nodes, 2, max_numpoints_region] float
                num_points_visual_regions_edges = inputs['num_points_visual_regions_edges']
                # num_points_visual_regions_edges: [batch_size, max_num_interacting_nodes] int

                # normalize visual regions with regards to batch padding
                visual_regions_edges = normalize_visual_regions(visual_regions_edges, image_shape,
                                                                pad_image_shape[1], pad_image_shape[2])
                # visual_regions_edges: [batch_size, max_num_interacting_nodes, 2, max_numpoints_region] float

                visual_edge_feature_list, \
                visual_edge_feature_dim_list = \
                    assign_visual_edge_features(feature_maps, visual_regions_edges, num_points_visual_regions_edges,
                                                self._feature_map_generation_params["layer_compressed_dim"],
                                                is_training,
                                                dropout_feature_map=self.graph_params["dropout_feature_map"],
                                                dropout_visual_feature_compression=self.graph_params[
                                                    "dropout_visual_feature_compression"])
                # visual_edge_feature_list: list of [batch_size, max_num_interacting_nodes, feature_map_i_channels] float

                visual_edge_features = tf.concat(visual_edge_feature_list, axis=-1)
                visual_edge_feature_dim = sum(visual_edge_feature_dim_list)
                # visual_edge_features: [batch_size, max_num_interacting_nodes, visual_feature_dim] float

                # add the visual features to the edge features
                if 'edge_features' in gnn_inputs:
                    gnn_inputs['edge_features'] = tf.concat([gnn_inputs['edge_features'], visual_edge_features],
                                                            axis=-1)
                else:
                    gnn_inputs['edge_features'] = gnn_inputs['edge_features']

        # optionally perform DropEdge
        if self.graph_params['dropout_edges']:
            gnn_inputs['interacting_nodes'] = drop_edge(gnn_inputs['interacting_nodes'], is_training,
                                                        rate=self.graph_params['dropout_edges'])

        # perform GNN update
        # GraphLSTM - Layer 1
        with tf.compat.v1.variable_scope('GraphLSTM1'):
            gnn_output_node_features = self._gnn1.infer(gnn_inputs, is_training)
        # here possibly more layers ...

        gnn_final = gnn_output_node_features['gnn_node_features']
        # gnn_final: [batch_size, max_num_nodes, output_node_feature_dim] float or None

        # If we are not using the GNN we directly classify on the node_features
        if gnn_final is None:
            gnn_final = inputs['node_features']
            # gnn_final: [batch_size, max_num_nodes, node_feature_dim] float

        # ===== classification
        with tf.compat.v1.variable_scope('Classification'):
            num_classes = self._flags.num_classes
            num_hidden_units = [int(s) for s in self._flags.num_hidden_units.split(',')]
            rel_to_consider = inputs['relations_to_consider_belong_to_same_instance']
            logits = self._get_only_logits_for_samples(rel_to_consider,
                                                       gnn_final,
                                                       is_training,
                                                       num_hidden_units,
                                                       num_classes,
                                                       "logits")
        return {'logits': logits}
            # logits_per_classifier = {}
            #
            # # for each classifier ...
            # for idx, classifier_name in enumerate(self._flags.classifiers):
            #
            #     # number of classes
            #     num_classes = self._flags.num_classes_per_classifier[idx]
            #     num_hidden_units = []
            #     num_hidden_units_str = self._flags.num_hidden_units_per_classifier[idx]
            #     num_hidden_list = num_hidden_units_str.split(",")
            #     for a_num_hidden in num_hidden_list:
            #         num_hidden_units.append(int(a_num_hidden))
            #     rel_to_consider = inputs['relations_to_consider'][classifier_name]
            #
            #     logits_per_classifier[classifier_name] = self._get_only_logits_for_samples(rel_to_consider,
            #                                                                                gnn_final,
            #                                                                                is_training,
            #                                                                                num_hidden_units,
            #                                                                                num_classes, classifier_name)
            # logits_per_classifier[classifier_name]: [batch_size, max_num_relations_to_consider, num_classes] float

        # logits_per_classifier: dictionary of [batch_size, max_num_relations_to_consider, num_classes] float
        # return {'logits': logits_per_classifier}

    def _get_only_logits_for_samples(self,
                                     rel_to_consider,
                                     gnn_node_features,
                                     is_training,
                                     num_hidden_units,
                                     num_classes,
                                     name):

        # rel_to_consider: [batch_size, max_num_relations_to_consider, num_relation_components] int
        # gnn_node_features: [batch_size, max_num_nodes, output_node_feature_dim] float
        # is_training: bool
        # num_hidden_units: [num_hidden_layers] int
        # num_classes:  int  number of classes including the no-relation class as 0
        # name: str  name of the classifier

        # TODO do it chunkwise!
        batch_size = tf.shape(rel_to_consider)[0]
        max_num_relations_to_consider = tf.shape(rel_to_consider)[1]
        num_relation_components = rel_to_consider.get_shape().as_list()[2]
        gnn_node_features_dim = gnn_node_features.get_shape().as_list()[2]

        # ===== collect the node features for each valid relation
        # gnn_node_features: [batch_size, max_num_nodes, gnn_node_feature_dim] float
        # rel_to_consider: [batch_size, max_num_relations_to_consider, num_relation_components] int
        relation_component_features = self._collect_features(gnn_node_features, rel_to_consider)
        # relation_component_features: [batch_size, max_num_relations_to_consider, num_relation_components, gnn_node_feature_dim] float

        # ===== combine the relation component features to the relation features
        relation_features = tf.reshape(relation_component_features,
                                       [batch_size * max_num_relations_to_consider,
                                        num_relation_components * gnn_node_features_dim])
        # relation_features: [batch_size * max_num_relations_to_consider, num_relation_components * gnn_node_features_dim] float

        # ===== classify the relations
        logits = tf.reshape(mlp(relation_features, num_hidden_units, num_classes, is_training=is_training,
                                dropout_rate=self.graph_params["dropout_classifier"], dropout_output=False,
                                reuse=tf.compat.v1.AUTO_REUSE, name=name),
                            [batch_size, max_num_relations_to_consider, num_classes])

        # logits: [batch_size, max_num_relations_to_consider, num_classes] float
        return logits

    def _collect_features(self, node_features, node_tuples):
        # node_features: [batch_size, max_num_nodes, node_feature_dim] float
        # node_tuples: [batch_size, max_num_node_tuples, node_tuple_length] int
        batch_size = tf.shape(node_tuples)[0]
        max_num_node_tuples = tf.shape(node_tuples)[1]
        node_tuple_length = tf.shape(node_tuples)[2]

        batch_range = tf.expand_dims(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]),
                                             [1, max_num_node_tuples, node_tuple_length]), axis=-1)
        # batch_range: [batch_size, max_num_node_tuples, node_tuple_length, 1]
        batch_node_tuples = tf.concat([batch_range, tf.expand_dims(node_tuples, axis=-1)], axis=3)
        # batch_node_tuples: [batch_size, max_num_node_tuples, node_tuple_length, 2]

        node_tuple_features = tf.gather_nd(node_features, batch_node_tuples)
        # node_tuple_features: [batch_size, max_num_node_tuples, node_tuple_length, node_feature_dim]

        return node_tuple_features
#
#
# if __name__ == '__main__':
#     def collect_features(node_features, node_tuples):
#         # node_features: [batch_size, max_num_nodes, node_feature_dim] float
#         # node_tuples: [batch_size, max_num_node_tuples, node_tuple_length] int
#         batch_size = tf.shape(node_tuples)[0]
#         max_num_node_tuples = tf.shape(node_tuples)[1]
#         node_tuple_length = tf.shape(node_tuples)[2]
#
#         batch_range = tf.expand_dims(tf.tile(tf.reshape(tf.range(batch_size), [-1, 1, 1]),
#                                              [1, max_num_node_tuples, node_tuple_length]), axis=-1)
#
#         # batch_range: [batch_size, max_num_node_tuples, node_tuple_length, 1]
#         batch_range = tf.Print(batch_range, [tf.shape(batch_range), batch_range], message="batch-range", summarize=1000)
#         batch_node_tuples = tf.concat([batch_range, tf.expand_dims(node_tuples, axis=-1)], axis=-1)
#         # batch_node_tuples: [batch_size, max_num_node_tuples, node_tuple_length, 2]
#
#         node_tuple_features = tf.gather_nd(node_features, batch_node_tuples)
#         # node_tuple_features: [batch_size, max_num_node_tuples, node_tuple_length, node_feature_dim]
#
#         return batch_node_tuples, node_tuple_features
#
#
#     nf = tf.constant([[[0.1, 0.2, 0.3, 0.4, 0.5],
#                        [0.1, 0.2, 0.3, 0.4, 0],
#                        [0.1, 0.2, 0.3, 0, 0]],
#                       [[-0.1, -0.2, -0.3, -0.4, -0.5],
#                        [-0.1, -0.2, -0.3, -0.4, 0],
#                        [0, 0, 0, 0, 0]]])
#
#     nt = tf.constant([[[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]],
#                       [[0, 0], [0, 1], [0, 2], [1, 0], [2, 0]]])
#
#     sess = tf.Session()
#
#     print(sess.run(tf.shape(nf)))
#     print(sess.run(tf.shape(nt)))
#
#     bnt, ntf = collect_features(nf, nt)
#
#     relation_features = tf.reshape(ntf,
#                                    [2 * 5,
#                                     2 * 5])
#
#     print(sess.run(tf.shape(bnt)))
#     print(sess.run(bnt))
#     print(sess.run(tf.shape(ntf)))
#     print(sess.run(ntf))
#     print(sess.run(tf.shape(relation_features)))
#     print(sess.run(relation_features))
