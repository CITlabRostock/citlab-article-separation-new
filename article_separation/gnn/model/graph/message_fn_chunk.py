import logging
import tensorflow as tf
from gnn.model.model_base import GraphBase
from gnn.model.graph_util import layers
from utils.flags import update_params


class MessageFnChunk(GraphBase):
    def __init__(self, params, message_fn_params):
        super(MessageFnChunk, self).__init__(params)

        # Default configuration for the message_fn graph
        self.graph_params["message_fct"] = 'default'  # GNN message function
        self.graph_params["num_interactions_per_chunk"] = 100000  # rough number of interactions calculated per chunk

        self.graph_params["aggregation_type"] = 'sum'  # which aggregation function to use in GNNs
        self.graph_params["interaction_feature_dim"] = 32  # interaction feature dimension
        self.graph_params["dropout_interaction"] = 0.0
        self.graph_params["dropout_interaction_output"] = False
        self.graph_params["dropout_attention"] = 0.0
        self.graph_params["dropout_attention_output"] = False

        # space seperated list of the number of units in the hidden layer for the interaction function mlp
        self.graph_params["num_hidden_units_interaction_fct"] = [32]

        # self.graph_params["incorporate_local_input_features_interaction"] = True  # incorporate the local node input features in the calculation of the interaction features
        # self.graph_params["incorporate_local_hidden_features_interaction"] = True  # incorporate the local node hidden features in the calculation of the interaction features
        # self.graph_params["incorporate_neighbor_input_features_interaction"] = True  # incorporate the input features of the neighbor node in the calculation of the interaction features
        # self.graph_params["incorporate_neighbor_hidden_features_interaction"] = True  # incorporate the hidden features of the neighbor node in the calculation of the interaction features
        # self.graph_params["incorporate_input_feature_difference_interaction"] = True  # incorporate the difference between local node input features and neighbor node input features in the calculation of the interaction features
        # self.graph_params["incorporate_hidden_feature_difference_interaction"] = True  # incorporate the difference between local node hidden features and neighbor node hidden features in the calculation of the interaction features
        # self.graph_params["incorporate_input_feature_squared_difference_interaction"] = True  # incorporate the squared difference between local node input features and neighbor node input features in the calculation of the interaction features
        # self.graph_params["incorporate_hidden_feature_squared_difference_interaction"] = True  # incorporate the squared difference between local node hidden features and neighbor node hidden features in the calculation of the interaction features

        self.graph_params["use_attention"] = False  # use attention or not
        # The following block is just of interest in case of usage of attention
        self.graph_params["num_attention_heads"] = 1  # number of attention heads
        self.graph_params["multihead_attention_merge_type"] = 'concat'  # merge type from multiple attention heads
        # space seperated list of the number of units in the hidden layer for the attention function mlp
        self.graph_params["num_hidden_units_attention_fct"] = [16]

        # self.graph_params["incorporate_local_input_features_attention"] = True  # incorporate the local node input features in the calculation of the interaction features and the attention values
        # self.graph_params["incorporate_local_hidden_features_attention"] = True  # incorporate the local node hidden features in the calculation of the interaction features and the attention values
        # self.graph_params["incorporate_neighbor_input_features_attention"] = True  # incorporate the input features of the neighbor node in the calculation of the attention values
        # self.graph_params["incorporate_neighbor_hidden_features_attention"] = True  # incorporate the hidden features of the neighbor node in the calculation of the attention values
        # self.graph_params["incorporate_input_feature_difference_attention"] = True  # incorporate the difference between local node input features and neighbor node input features in the calculation of the attention values
        # self.graph_params["incorporate_hidden_feature_difference_attention"] = True  # incorporate the difference between local node hidden features and neighbor node hidden features in the calculation of the attention values
        # self.graph_params["incorporate_input_feature_squared_difference_attention"] = True  # incorporate the squared difference between local node input features and neighbor node input features in the calculation of the attention values
        # self.graph_params["incorporate_hidden_feature_squared_difference_attention"] = True  # incorporate the squared difference between local node hidden features and neighbor node hidden features in the calculation of the attention_values

        # Updating of the default params if provided via flags as a dict
        self.graph_params = update_params(self.graph_params, message_fn_params, "Message_fn")

        if self.graph_params["message_fct"] == 'default':
            self._message_fct = self._message_function_default

        if self.graph_params["aggregation_type"] == 'sum':
            self._aggregation_fct = tf.sparse.reduce_sum
        elif self.graph_params["aggregation_type"] == 'max':
            self._aggregation_fct = tf.sparse.reduce_max
        else:
            self._aggregation_fct = None

        self._x_dim = self.graph_params["interaction_feature_dim"]
        if self.graph_params["use_attention"]:
            if self.graph_params["multihead_attention_merge_type"] == 'concat':
                self._x_dim = self._x_dim // self.graph_params["num_attention_heads"]

    def print_params(self):
        sorted_dict = sorted(self.graph_params.items(), key=lambda kv: kv[0])
        if len(sorted_dict) > 0:
            logging.info("message_fn_params:")
            for a in sorted_dict:
                logging.info(f"  {a[0]}: {a[1]}")

    def infer(self, inputs, is_training):
        interaction_chunk_size = tf.math.maximum(1, self.graph_params["num_interactions_per_chunk"] //
                                                 inputs['max_num_nodes_per_sample'])

        # Initialization of the node features.
        node_in_batch_interaction_features = tf.zeros(
            [inputs['num_nodes_in_batch'], self.graph_params["interaction_feature_dim"]],
            dtype=tf.float32)

        # node_in_batch_interaction_features: [num_nodes_in_batch, x_dim] float

        def message_function_single_chunk(current_node_in_batch_c, node_in_batch_interaction_features_c):
            # TODO make sure (via a second while loop) that the chunks are more or less equally sized / maybe not important
            interacting_nodes_in_batch_indices_c = self._get_interaction_chunk_indices(
                inputs['interacting_nodes_in_batch'],
                current_node_in_batch_c,
                tf.math.minimum(interaction_chunk_size, inputs['num_nodes_in_batch'] - current_node_in_batch_c))
            interacting_nodes_in_batch_c = tf.gather(inputs['interacting_nodes_in_batch'],
                                                     interacting_nodes_in_batch_indices_c)
            # interacting_nodes_in_batch_c: [num_interactions_in_chunk, 2] ; num_interactions_in_chunk <<chunk_size (typically?!)
            edge_features_in_batch_c = None
            if 'edge_features_in_batch' in inputs:
                edge_features_in_batch_c = tf.gather(inputs['edge_features_in_batch'],
                                                     interacting_nodes_in_batch_indices_c)
                # edge_features_in_batch_c: [num_interactions_in_chunk, dim]
            # ===== message function
            x_chunk = self._message_fct(inputs, is_training, interacting_nodes_in_batch_c, edge_features_in_batch_c)
            # x_chunk: [num_nodes_in_batch, x_dim] float

            node_in_batch_interaction_features_c += x_chunk

            current_node_c = current_node_in_batch_c + interaction_chunk_size

            return [current_node_c, node_in_batch_interaction_features_c]

        current_node = tf.constant(0, dtype=tf.int32)
        _, node_in_batch_interaction_features = tf.while_loop(
            lambda current_node, node_interaction_feature: tf.less(current_node, inputs['num_nodes_in_batch']),
            message_function_single_chunk,
            loop_vars=[current_node, node_in_batch_interaction_features],
            shape_invariants=[current_node.get_shape(), node_in_batch_interaction_features.get_shape()],
            swap_memory=True)

        return {'x': node_in_batch_interaction_features}

    def _get_interaction_chunk_indices(self, interacting_nodes_in_batch, start_node_in_batch, chunk_size):
        """
        Looks for interaction_indices whose end node is withing the current chunk
        """
        # interacting_nodes: [num_interactions_in_batch, 2] int
        # start_node: scalar int
        # chunk_size: scalar int
        end_node_in_batch = start_node_in_batch + chunk_size - 1
        node_in_batch_indices = self._get_interaction_node_index(interacting_nodes_in_batch)
        # node_in_batch_indices: [num_interactions_in_batch] int
        interaction_chunk_indices = tf.reshape(tf.where(
            tf.logical_and(tf.greater_equal(node_in_batch_indices, start_node_in_batch),
                           tf.less_equal(node_in_batch_indices, end_node_in_batch))), [-1])
        # interaction_chunk_mask: [<<chunk_size, 2] bool
        return interaction_chunk_indices

    def _get_interaction_node_index(self, interactions_to_use):
        """
        Slices all start node indices off of the interactions
        """
        # interacting_nodes_in_batch:
        num_interactions = tf.shape(interactions_to_use)[0]

        node_indices = tf.reshape(tf.slice(interactions_to_use, [0, 1], [num_interactions, 1]), [-1])
        # node_indices: [num_interactions] int
        return node_indices

    def _message_function_default(self, inputs, is_training, interacting_nodes_in_batch_to_use=None,
                                  edge_features_in_batch_to_use=None):

        num_nodes_in_batch = inputs['num_nodes_in_batch']

        # Whether to calc it full or chunkwise
        if interacting_nodes_in_batch_to_use is not None:
            interactions_to_use = interacting_nodes_in_batch_to_use
        else:
            interactions_to_use = inputs['interacting_nodes_in_batch']
        # interactions_to_use: [num_interactions, 2] int
        if edge_features_in_batch_to_use is not None:
            edge_features_to_use = edge_features_in_batch_to_use
        elif 'edge_features_in_batch' in inputs:
            edge_features_to_use = inputs['edge_features_in_batch']
        else:
            edge_features_to_use = None
        # edge_features_to_use: [num_interactions, dim] float

        with tf.compat.v1.variable_scope('message_fn_default'):
            num_attention_heads = self.graph_params["num_attention_heads"]
            if not self.graph_params["use_attention"]:
                num_attention_heads = 1

            aggregated_interaction_features = []

            for attention_head in range(num_attention_heads):
                with tf.compat.v1.variable_scope('head_' + str(attention_head)):

                    # ===== calculate interaction features
                    # gather features and calculate interaction features from them

                    # config=[self.graph_params["incorporate_local_input_features_interaction"],
                    #         self.graph_params["incorporate_local_hidden_features_interaction"],
                    #         self.graph_params["incorporate_neighbor_input_features_interaction"],
                    #         self.graph_params["incorporate_neighbor_hidden_features_interaction"],
                    #         self.graph_params["incorporate_input_feature_difference_interaction"],
                    #         self.graph_params["incorporate_hidden_feature_difference_interaction"],
                    #         self.graph_params["incorporate_input_feature_squared_difference_interaction"],
                    #         self.graph_params["incorporate_hidden_feature_squared_difference_interaction"]
                    #         ]
                    config = [True, True, True, True, True, True, True, True]
                    num_hidden_units = self.graph_params["num_hidden_units_interaction_fct"]
                    interaction_features = self._get_interaction_features(inputs, is_training, interactions_to_use,
                                                                          edge_features_to_use, config=config,
                                                                          num_hidden_units=num_hidden_units,
                                                                          feat_dim=self._x_dim,
                                                                          output_activation=layers.tanh,
                                                                          dropout_rate=self.graph_params[
                                                                              "dropout_interaction"],
                                                                          dropout_output=self.graph_params[
                                                                              "dropout_interaction_output"])
                    # interaction_features: [num_interactions, x_dim] float

                    # neighbor weighting
                    if self.graph_params["use_attention"]:
                        # gather features and calculate an attention value from them
                        unnormalized_attention_values = self._get_unnormalized_attention_values(inputs, is_training,
                                                                                                interactions_to_use,
                                                                                                edge_features_to_use)
                        # unnormalized_attention_values: [num_interactions] float
                        unnormalized_attention_tensor = tf.sparse.reorder(
                            tf.SparseTensor(indices=tf.cast(interactions_to_use, dtype=tf.int64),
                                            values=tf.cast(unnormalized_attention_values, dtype=tf.float32),
                                            dense_shape=[num_nodes_in_batch, num_nodes_in_batch]))
                        # unnormalized_attention_tensor: sparse [num_nodes, num_nodes] float
                        attention_values = self._softmax_normalize_attention_tensor(
                            unnormalized_attention_tensor).values
                        # attention_values: [num_interactions] float
                    else:
                        # balanced contribution of each neighbor node (normalized by the node degree)
                        attention_values = self._get_balanced_attention_values(interactions_to_use, num_nodes_in_batch)
                        # attention_values: [num_interactions] float

                    # ===== attenuate interaction features
                    # interaction_features: [num_interactions, x_dim] float
                    # attention_values: [num_interactions] float
                    attenuated_interaction_features = interaction_features * tf.expand_dims(attention_values, axis=-1)
                    # attenuated_interaction_features: [num_interactions, x_dim] float

                    # ===== aggregate interaction features
                    # interacting_nodes: [num_interactions, 2] int
                    # num_nodes: scalar int
                    aggregated_interaction_features.append(
                        self._aggregate_interaction_features(interactions_to_use, attenuated_interaction_features,
                                                             num_nodes_in_batch))
                    # aggregated_interaction_features[attention_head]: [num_nodes, x_dim] float

            # ====== combine the results of the multiple attention heads
            node_interaction_feature = None
            if not self.graph_params["use_attention"] or self.graph_params[
                "multihead_attention_merge_type"] == 'average':
                # aggregated_interaction_features: list of [num_nodes, x_dim] float and length num_attention_heads
                node_interaction_feature = tf.add_n(aggregated_interaction_features) / num_attention_heads
                # node_interaction_features: [num_nodes, x_dim] float
            elif self.graph_params["multihead_attention_merge_type"] == 'concat':
                # aggregated_interaction_features: list of [num_nodes, x_dim] float and length num_attention_heads
                node_interaction_feature = tf.concat(aggregated_interaction_features, axis=-1)
                # node_interaction_features: [num_nodes, num_attention_heads * x_dim] float

            return node_interaction_feature

    def _get_interaction_features(self, inputs, is_training, interactions_to_use, edge_features_to_use, config,
                                  num_hidden_units, feat_dim, output_activation, dropout_rate=0.0,
                                  dropout_output=False):
        with tf.compat.v1.variable_scope('calculation_interaction_features'):
            feature_dict = self._gather_features_from_interacting_node_indices_1(inputs, interactions_to_use, config)
            if edge_features_to_use is not None:
                feature_dict['edge_features_to_use'] = edge_features_to_use

            interaction_features = self._calculate_interaction_features_1(inputs, is_training, interactions_to_use,
                                                                          feature_dict, config, num_hidden_units,
                                                                          feat_dim, output_activation,
                                                                          dropout_rate, dropout_output)
            # interaction_features: [num_interactions, x_dim] float

            return interaction_features

    def _gather_features_from_interacting_node_indices_1(self, inputs, interactions_to_use, config):

        results = dict()
        if 'node_input_features_in_batch' in inputs:
            if config[0] or config[2] or config[4] or config[6]:
                u_from, u_to = self._gather_node_features(inputs['node_input_features_in_batch'], interactions_to_use)
                if config[0]:
                    results['u_from'] = u_from
                if config[2]:
                    results['u_to'] = u_to
                if config[4]:
                    results['u_diff'] = u_to - u_from
                if config[6]:
                    results['u_squared_diff'] = (u_to - u_from) ** 2
        elif config[0] or config[2] or config[4] or config[6]:
            logging.warning("Message function wants to use node_input_features. These are NOT available.")

        if config[1] or config[3] or config[5] or config[7]:
            h_from, h_to = self._gather_node_features(inputs['node_hidden_features_in_batch'], interactions_to_use)
            if config[1]:
                results['h_from'] = h_from
            if config[3]:
                results['h_to'] = h_to
            if config[5]:
                results['h_diff'] = h_to - h_from
            if config[7]:
                results['h_squared_diff'] = (h_to - h_from) ** 2

        return results

    def _gather_node_features(self, node_features_in_batch, interactions_to_use):
        # node_features_in_batch: [bS*max_nodes, node_feature_dim] float
        # interacting_nodes: [num_interactions, 2] int
        node_feature_from, node_feature_to = tf.unstack(tf.gather(node_features_in_batch, interactions_to_use), axis=1)
        # node_feature_from: [num_interactions, node_feature_dim] float
        # node_feature_to: [num_interactions, node_feature_dim] float
        return node_feature_from, node_feature_to

    def _calculate_interaction_features_1(self, inputs, is_training, interactions_to_use, feature_dict, config,
                                          num_hidden_units, feat_dim, output_activation,
                                          dropout_rate=0.0, dropout_output=False):

        # interacting_nodes: [num_interactions, 2] int  indices of interacting nodes which are of interest
        num_interactions = tf.shape(interactions_to_use)[0]

        u_features = tf.zeros([num_interactions, 0], dtype=tf.float32)
        if 'node_input_features_in_batch' in inputs:
            if config[0]:
                # feature_dict['u_from']: [num_interactions, u_node_dim] float
                u_features = tf.concat([u_features, feature_dict['u_from']], axis=-1)

            if config[2]:
                # feature_dict['u_to']: [num_interactions, u_node_dim] float
                u_features = tf.concat([u_features, feature_dict['u_to']], axis=-1)

            if config[4]:
                # feature_dict['u_diff']: [num_interactions, u_node_dim] float
                u_features = tf.concat([u_features, feature_dict['u_diff']], axis=-1)

            if config[6]:
                # feature_dict['u_squared_diff']: [num_interactions, u_node_dim] float
                u_features = tf.concat([u_features, feature_dict['u_squared_diff']], axis=-1)

        if 'edge_features_to_use' in feature_dict:
            # feature_dict['edge_features_to_use']: [num_interactions, u_edge_dim] float
            u_features = tf.concat([u_features, feature_dict['edge_features_to_use']], axis=-1)

        # u_features: [num_interactions, u_feature_dim] float

        h_features = tf.zeros([num_interactions, 0], dtype=tf.float32)
        if config[1]:
            # feature_dict['h_from']: [num_interactions, h_dim] float
            h_features = tf.concat([h_features, feature_dict['h_from']], axis=-1)

        if config[3]:
            # feature_dict['h_to']: [num_interactions, h_dim] float
            h_features = tf.concat([h_features, feature_dict['h_to']], axis=-1)

        if config[5]:
            # feature_dict['h_diff']: [num_interactions, h_dim] float
            h_features = tf.concat([h_features, feature_dict['h_diff']], axis=-1)

        if config[7]:
            # feature_dict['h_squared_diff']: [num_interactions, h_dim] float
            h_features = tf.concat([h_features, feature_dict['h_squared_diff']], axis=-1)

        # h_features: [num_interactions, h_feature_dim] float

        # ===== combine u and h features

        with tf.compat.v1.variable_scope('concat_u_and_h'):
            # u_features: [num_interactions, u_feature_dim] float
            # h_features: [num_interactions, h_feature_dim] float
            interaction_features = layers.mlp(tf.concat([u_features, h_features], axis=-1), num_hidden_units, feat_dim,
                                              is_training, hidden_activation=layers.relu,
                                              output_activation=output_activation,
                                              use_bias=True, dropout_rate=dropout_rate, dropout_output=dropout_output,
                                              reuse=tf.compat.v1.AUTO_REUSE, name='interaction_features')

        # interaction_features: [num_interactions, feat_dim] float

        return interaction_features

    def _get_balanced_attention_values(self, interactions_to_use, num_nodes_in_batch):
        # interacting_nodes: [num_interactions, 2] int
        # num_nodes_in_batch: scalar int
        interaction_node_index = self._get_interaction_node_index(interactions_to_use)
        # interaction_node_index: [num_interactions] int

        interaction_tensor = self._get_interaction_tensor(interactions_to_use, num_nodes_in_batch)
        # interaction_tensor: sparse [num_nodes_in_batch, num_nodes_in_batch] float (integer as float)
        node_degree = tf.sparse.reduce_sum(interaction_tensor, axis=0)
        # the following line of code is necessary, since otherwise the shape of node_degree is unknown!!
        node_degree = tf.reshape(node_degree, [num_nodes_in_batch])
        # node_degree: [num_nodes] float    The number of INCOMING edges per node
        # interaction_node_index: [num_interactions] int
        interacting_nodes_degree = tf.gather(node_degree, interaction_node_index)
        # interacting_nodes_degree: [num_interactions] float
        balanced_attention_values = 1.0 / interacting_nodes_degree
        # balanced_attention_values: [num_interactions] float
        return balanced_attention_values

    def _get_interaction_tensor(self, interactions_to_use, num_nodes_in_batch):
        # interactions_to_use: [num_interactions, 2] int
        # num_nodes_in_batch: scalar int
        num_interactions = tf.shape(interactions_to_use)[0]
        interaction_tensor = tf.sparse.reorder(tf.SparseTensor(indices=tf.cast(interactions_to_use, dtype=tf.int64),
                                                               values=tf.ones([num_interactions], dtype=tf.float32),
                                                               dense_shape=[num_nodes_in_batch, num_nodes_in_batch]))
        # interaction_tensor: sparse [num_nodes, num_nodes] float
        return interaction_tensor

    def _aggregate_interaction_features(self, interactions_to_use, interaction_features, num_nodes_in_batch):
        # interactions_to_use: [num_interactions, 2] int
        # interaction_features: [num_interactions, x_dim] float
        # num_nodes_in_batch: scalar int
        def _aggregate_single_feature_component(interaction_features_single_component):
            # interaction_features_single_component: [num_interactions] float
            aggregated_single_feature_component = self._aggregation_fct(
                tf.sparse.reorder(tf.SparseTensor(indices=tf.cast(interactions_to_use, dtype=tf.int64),
                                                  values=tf.cast(interaction_features_single_component,
                                                                 dtype=tf.float32),
                                                  dense_shape=[num_nodes_in_batch, num_nodes_in_batch])), axis=0)
            # the following line of code is necessary, since otherwise the shape of aggregated_single_feature_component is unknown!!
            aggregated_single_feature_component = tf.reshape(aggregated_single_feature_component, [num_nodes_in_batch])
            # aggregated_single_feature_component: [num_nodes] float
            return aggregated_single_feature_component

        aggregated_interaction_features = tf.transpose(
            tf.map_fn(_aggregate_single_feature_component, tf.transpose(interaction_features, perm=[1, 0]),
                      dtype=tf.float32), perm=[1, 0])
        # aggregated_interaction_features: [num_nodes_in_batch, x_dim] float
        return aggregated_interaction_features

    def _get_unnormalized_attention_values(self, inputs, is_training, interactions_to_use, edge_features_to_use):

        with tf.compat.v1.variable_scope('calculation_unnormalized_attention_values'):
            # config=[self.graph_params["incorporate_local_input_features_attention"],
            #         self.graph_params["incorporate_local_hidden_features_attention"],
            #         self.graph_params["incorporate_neighbor_input_features_attention"],
            #         self.graph_params["incorporate_neighbor_hidden_features_attention"],
            #         self.graph_params["incorporate_input_feature_difference_attention"],
            #         self.graph_params["incorporate_hidden_feature_difference_attention"],
            #         self.graph_params["incorporate_input_feature_squared_difference_attention"],
            #         self.graph_params["incorporate_hidden_feature_squared_difference_attention"]
            #         ]
            config = [True, True, True, True, True, True, True, True]
            num_hidden_units = self.graph_params["num_hidden_units_attention_fct"]
            unnormalized_attention_values = self._get_interaction_features(inputs, is_training, interactions_to_use,
                                                                           edge_features_to_use, config=config,
                                                                           num_hidden_units=num_hidden_units,
                                                                           feat_dim=1,
                                                                           output_activation=None,
                                                                           dropout_rate=self.graph_params[
                                                                               "dropout_attention"],
                                                                           dropout_output=self.graph_params[
                                                                               "dropout_attention_output"])
            # unnormalized_attention_values: [num_interactions, 1] float
            unnormalized_attention_values = tf.squeeze(unnormalized_attention_values, axis=-1)
            # unnormalized_attention_values: [num_interactions] float

            return unnormalized_attention_values

    def _softmax_normalize_attention_tensor(self, attention_tensor):
        # attention_tensor: sparse [num_nodes_in_batch, num_nodes_in_batch] float
        normalized_attention_tensor = tf.sparse.softmax(tf.sparse.transpose(attention_tensor, perm=[1, 0]))
        # normalized_attention_tensor: sparse [num_nodes_in_batch, num_nodes_in_batch] float
        return normalized_attention_tensor
#
#
# if __name__ == '__main__':
#     msg_function = MessageFnChunk({'flags': None}, {})
#
#     inputs = dict()
#     # inputs['num_nodes_in_batch']            #  [] int (batch_size * max_num_nodes)
#     inputs['num_nodes_in_batch'] = tf.constant(4, dtype=tf.int32)
#     # inputs['max_num_nodes_per_sample']      #  [] int
#     inputs['max_num_nodes_per_sample'] = tf.constant(4, dtype=tf.int32)
#     # inputs['interacting_nodes_in_batch']    #  [num_interactions_in_batch, 2] int
#     inputs['interacting_nodes_in_batch'] = tf.constant([[0, 1], [0, 2], [0, 3],
#                                                         [1, 0], [1, 2],
#                                                         [2, 0], [2, 1],
#                                                         [3, 0]], dtype=tf.int32)
#     # inputs['node_hidden_features_in_batch'] #  [batch_size * max_num_nodes, hidden_node_feature_dim] float
#     inputs['node_hidden_features_in_batch'] = tf.constant([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
#                                                           dtype=tf.float32)
#     # inputs['edge_features_in_batch']        #  [num_interactions_in_batch, edge_feature_dim] float
#     inputs['edge_features_in_batch'] = tf.constant([[0, 1, 0, 1], [0, 2, 0, 2], [0, 3, 0, 3],
#                                                     [1, 0, 1, 0], [1, 2, 1, 2],
#                                                     [2, 0, 2, 0], [2, 1, 2, 1],
#                                                     [3, 0, 3, 0]], dtype=tf.float32)
#     # inputs['node_input_features_in_batch']  #  [batch_size * max_num_nodes, node_feature_dim] float
#     inputs['node_input_features_in_batch'] = tf.constant([[0, 0], [10, 10], [20, 20], [30, 30]], dtype=tf.float32)
#
#     msg = msg_function.infer(inputs, is_training=True)
#     sess = tf.Session()
#     sess.run(tf.compat.v1.global_variables_initializer())
#     m = sess.run(msg)
