import logging
import tensorflow as tf
from gnn.model.model_base import GraphBase
from gnn.model.graph.message_fn_chunk import MessageFnChunk
from gnn.model.graph.update_fn_lstm import UpdateFnLSTM
from gnn.model.graph_util.misc import check_and_correct_interacting_nodes
from gnn.model.graph_util import layers
from utils.flags import update_params


class GraphGNN(GraphBase):
    def __init__(self, params, gnn_params, message_fn_params, update_fn_params):
        """
        :param params:
        """
        super(GraphGNN, self).__init__(params)

        # Default configuration for the gnn graph
        self.graph_params["num_transition_steps"] = 3  # 'Timesteps' of the GNN. If this is set to 0, we return None.
        self.graph_params["compress_node_feature_dim"] = 0  # Hidden dim of this FF layer (disabled for value of 0)
        self.graph_params["dropout_rate_node_features"] = 0.0  # self explanatory
        self.graph_params["undirected_graph"] = True  # self explanatory
        self.graph_params["output_type"] = 'hidden'  # 'hidden', 'add/concat_final_hidden_and_input'
        self.graph_params["message_fct"] = 'CHUNK'  # GNN message function
        self.graph_params["update_fct"] = 'LSTM'  # GNN update function

        # Updating of the default params if provided via flags as a dict
        self.graph_params = update_params(self.graph_params, gnn_params, "GNN")

        self._graph_message_fct = None
        if self.graph_params["message_fct"] == 'CHUNK':
            self._graph_message_fct = MessageFnChunk(params, message_fn_params)
        self._graph_update_fct = None
        if self.graph_params["update_fct"] == 'LSTM':
            self._graph_update_fct = UpdateFnLSTM(params, update_fn_params)

    def print_params(self):
        sorted_dict = sorted(self.graph_params.items(), key=lambda kv: kv[0])
        if len(sorted_dict) > 0:
            logging.info("gnn_params:")
            for a in sorted_dict:
                logging.info(f"  {a[0]}: {a[1]}")
        self._graph_message_fct.print_params()
        self._graph_update_fct.print_params()

    def infer(self, inputs, is_training):
        """
        :param inputs: Dict containing num_nodes, interacting_nodes, num_interacting_nodes,
                       node_features (optional), edge_features (optional)
        :param is_training:
        :return:
        """
        if self.graph_params["num_transition_steps"] == 0:
            return {'gnn_node_features': None}

        edge_features = inputs['edge_features'] if 'edge_features' in inputs else None
        # Correction of the input
        corrected_interacting_nodes, corrected_edge_features, corrected_num_interacting_nodes = \
            check_and_correct_interacting_nodes(inputs['interacting_nodes'],
                                                edge_features,
                                                inputs['num_nodes'],
                                                inputs['num_interacting_nodes'],
                                                self.graph_params["undirected_graph"])

        # Reformulation from [batch, ...] to a flat structure, this is meaningful
        # due to the fact, that we have separable graph structures
        gnn_input_final = dict()
        num_nodes = inputs['num_nodes']
        max_num_nodes = tf.reduce_max(num_nodes)  # [] int
        batch_size = tf.shape(num_nodes)[0]
        num_nodes_in_batch = batch_size * max_num_nodes
        gnn_input_final['num_nodes_in_batch'] = num_nodes_in_batch
        gnn_input_final['max_num_nodes_per_sample'] = max_num_nodes
        interacting_nodes_mask = tf.sequence_mask(
            corrected_num_interacting_nodes)  # [batch_size, max_num_interactions] bool
        interacting_nodes = corrected_interacting_nodes  # [batch_size, max_num_interacting_nodes, 2] int

        # ===== give the nodes unique indices over the whole batch
        # to handle all nodes of the examples in the batch withing a single graph
        # we have to give each node a unique index (using a batch_offset of batch_index * max_num_nodes)
        reindexed_interacting_nodes = self._reindex_nodes_over_batch(interacting_nodes, max_num_nodes)
        # reindexed_interacting_nodes: [batch_size, max_num_interactions, 2] int
        interacting_nodes_pos = tf.where(interacting_nodes_mask)
        # interacting_nodes_pos: [num_interactions_in_batch, 2] int
        interacting_nodes_in_batch = tf.gather_nd(reindexed_interacting_nodes, interacting_nodes_pos)
        # interacting_nodes_in_batch: [num_interactions_in_batch, 2] int

        gnn_input_final['interacting_nodes_in_batch'] = interacting_nodes_in_batch
        ##### interaction nodes DONE

        if 'edge_features' in inputs:
            # corrected_edge_features: [batch_size, corrected_max_num_interacting_nodes, edge_feature_dim] float
            # interacting_nodes_pos: [num_interactions_in_batch, 2] int
            edge_features_in_batch = tf.gather_nd(corrected_edge_features, interacting_nodes_pos)
            # edge_features_in_batch: [num_interactions_in_batch, edge_feature_dim] float
            gnn_input_final['edge_features_in_batch'] = edge_features_in_batch
        ##### edge features DONE

        if 'node_features' in inputs:
            node_features = inputs['node_features']  # [batch_size, max_num_nodes, node_feature_dim] float
            node_feature_dim = node_features.get_shape().as_list()[-1]
            if self.graph_params["compress_node_feature_dim"] > 0:
                with tf.compat.v1.variable_scope("compress_input"):
                    # node_features: [batch_size, max_num_nodes, node_feature_dim] float
                    node_features = layers.ff_layer(node_features, outD=self.graph_params["compress_node_feature_dim"],
                                                    is_training=is_training, activation=layers.tanh,
                                                    name='ff_compress_input')
                    # node_features: [batch_size, max_num_nodes, compress_node_feature_dim] float
                    node_feature_dim = self.graph_params["compress_node_feature_dim"]
            # ===== dropout
            if self.graph_params["dropout_rate_node_features"] > 0:
                node_features = layers.dropout(node_features, is_training, name="dropout_node_features",
                                               rate=self.graph_params["dropout_rate_node_features"])

            # Now ensure that the reindex is consistent to the node feature order
            # node_features: [batch_size, max_num_nodes, node_feature_dim] float
            node_input_features_in_batch = tf.reshape(node_features, [num_nodes_in_batch, node_feature_dim])
            # node_input_features_in_batch: [batch_size * max_num_nodes, node_feature_dim] float
            gnn_input_final['node_input_features_in_batch'] = node_input_features_in_batch
        ##### node features DONE

        # ===== GNN update

        # ===== initialize hidden and cell states with zero
        h = tf.zeros([num_nodes_in_batch, self._graph_update_fct.get_hidden_dim()])
        # h: [batch_size * max_num_nodes, hidden_node_feature_dim] float
        cellstate = tf.zeros([num_nodes_in_batch, self._graph_update_fct.get_hidden_dim()])
        # cellstate: [batch_size * max_num_nodes, hidden_node_feature_dim] float

        for transition_step in range(self.graph_params["num_transition_steps"]):
            # ===== message function
            input_message = dict()
            input_message['num_nodes_in_batch'] = gnn_input_final['num_nodes_in_batch']
            input_message['max_num_nodes_per_sample'] = gnn_input_final['max_num_nodes_per_sample']
            input_message['interacting_nodes_in_batch'] = gnn_input_final['interacting_nodes_in_batch']
            input_message['node_hidden_features_in_batch'] = h
            if 'edge_features_in_batch' in gnn_input_final:
                input_message['edge_features_in_batch'] = gnn_input_final['edge_features_in_batch']
            if 'node_input_features_in_batch' in gnn_input_final:
                input_message['node_input_features_in_batch'] = gnn_input_final['node_input_features_in_batch']
            message_out = self._graph_message_fct.infer(input_message, is_training)

            # ===== update function
            input_update = dict()
            input_update['cellstate'] = cellstate
            input_update['h'] = h
            input_update['x'] = message_out['x']
            if 'node_input_features_in_batch' in gnn_input_final:
                input_update['node_input_features_in_batch'] = gnn_input_final['node_input_features_in_batch']
            update_out = self._graph_update_fct.infer(input_update, is_training)
            h = update_out['h']
            cellstate = update_out['cellstate']

        # go back from flat structure to batch output
        out = tf.reshape(h, [batch_size, max_num_nodes, self._graph_update_fct.get_hidden_dim()])
        # out: [batch_size, max_num_nodes, hidden_node_feature_dim] float

        # optionally add or concat input features to output
        if 'node_features' in inputs:
            if self.graph_params["output_type"] == 'add_final_hidden_and_input':
                out += layers.ff_layer(inputs['node_features'], outD=self._graph_update_fct.get_hidden_dim(),
                                       activation=None,
                                       use_bias=False, is_training=is_training, reuse=tf.compat.v1.AUTO_REUSE)
            elif self.graph_params["output_type"] == 'concat_final_hidden_and_input':
                out = tf.concat([out, inputs['node_features']], axis=-1)

        return {'gnn_node_features': out}

    # def _reindex_nodes_over_batch(self, interacting_nodes, max_num_nodes):
    #     # interacting_nodes: [batch_size, max_num_interacting_nodes, 2] int
    #     # max_num_nodes: scalar int
    #     node_indices_shape = tf.shape(interacting_nodes)
    #     # node_indices_shape: [1 + remaining_dimensions] int
    #     rank = tf.shape(node_indices_shape)[0]
    #     # rank: scalar int
    #     permutation = tf.concat(
    #         [tf.expand_dims(rank - 1, axis=-1), tf.range(1, limit=rank - 1), tf.constant([0], dtype=tf.int32)], axis=0)
    #     # permuation: [1 + remaining_dimensions]
    #
    #     batch_size = node_indices_shape[0]
    #     offset = tf.range(0, limit=batch_size) * max_num_nodes
    #     # offset: [batch_size] int
    #
    #     new_interacting_nodes = tf.transpose(tf.transpose(interacting_nodes, perm=permutation) + offset,
    #                                          perm=permutation)
    #     # new_node_indices: [batch_size, max_num_interacting_nodes, 2] int
    #     return new_interacting_nodes

    def _reindex_nodes_over_batch(self, interacting_nodes, max_num_nodes):
        # interacting_nodes: [batch_size, max_num_interacting_nodes, 2] int
        # max_num_nodes: scalar int
        batch_size = tf.shape(interacting_nodes)[0]
        offset = tf.range(batch_size) * max_num_nodes
        # offset: [batch_size] int
        new_interacting_nodes = interacting_nodes + tf.reshape(offset, [-1, 1, 1])
        return new_interacting_nodes
#
#
# if __name__ == '__main__':
#     def reindex_nodes_over_batch(interacting_nodes, max_num_nodes):
#         # interacting_nodes: [batch_size, max_num_interacting_nodes, 2] int
#         # max_num_nodes: scalar int
#         node_indices_shape = tf.shape(interacting_nodes)
#         # node_indices_shape: [1 + remaining_dimensions] int
#         rank = tf.shape(node_indices_shape)[0]
#         rank = tf.Print(rank, [rank], message="rank ", summarize=999)
#         # rank: scalar int
#         permutation = tf.concat(
#             [tf.expand_dims(rank - 1, axis=-1), tf.range(1, limit=rank - 1), tf.constant([0], dtype=tf.int32)], axis=0)
#         permutation = tf.Print(permutation, [permutation], message="permutation ", summarize=999)
#         # permuation: [1 + remaining_dimensions]
#
#         batch_size = node_indices_shape[0]
#         offset = tf.range(0, limit=batch_size) * max_num_nodes
#         offset = tf.Print(offset, [offset], message="offset ", summarize=999)
#         # offset: [batch_size] int
#
#         new_interacting_nodes = tf.transpose(tf.transpose(interacting_nodes, perm=permutation) + offset,
#                                              perm=permutation)
#         # new_node_indices: [batch_size, max_num_interacting_nodes, 2] int
#         return new_interacting_nodes
#
#
#     def reindex_nodes_over_batch2(interacting_nodes, max_num_nodes):
#         batch_size = tf.shape(interacting_nodes)[0]
#         offset = tf.range(batch_size) * max_num_nodes
#         new_interacting_nodes = interacting_nodes + tf.reshape(offset, [-1, 1, 1])
#         return new_interacting_nodes
#
#
#     inter_nodes = tf.constant([[[0, 1], [1, 0], [0, 2], [2, 0], [1, 2], [2, 1]],
#                                [[0, 1], [1, 0], [1, 2], [2, 1], [1, 3], [3, 1]]])
#
#     max_nodes = 4
#     new_nodes = reindex_nodes_over_batch(inter_nodes, max_nodes)
#     new_nodes2 = reindex_nodes_over_batch2(inter_nodes, max_nodes)
#
#     sess = tf.Session()
#
#     n1, n2, n3 = sess.run([inter_nodes, new_nodes, new_nodes2])
#     # print(n1)
#     print("new1\n", n2)
#     print("new2\n", n3)
#
#     mask = tf.constant([[1, 1, 1, 1, 0, 0],
#                         [1, 1, 1, 1, 1, 1]])
#     nodes_pos = tf.where(mask)
#     nodes_in_batch = tf.gather_nd(new_nodes2, nodes_pos)
#
#     i2 = sess.run([nodes_in_batch])
#     print("i2\n", i2)
