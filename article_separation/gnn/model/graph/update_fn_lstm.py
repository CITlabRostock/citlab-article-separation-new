import logging
import tensorflow as tf
from gnn.model.model_base import GraphBase
from gnn.model.graph_util import layers
from utils.flags import update_params


class UpdateFnLSTM(GraphBase):
    def __init__(self, params, update_fn_params):
        super(UpdateFnLSTM, self).__init__(params)

        self.graph_params["hidden_node_feature_dim"] = 32  # hidden node feature dimension
        self.graph_params[
            "incorporate_hidden_features_in_update"] = True  # Use (or not) the hidden features as input for the current time step
        self.graph_params[
            "incorporate_node_input_features_in_update"] = True  # Use (or not) the node input features as input for the current time step

        self.graph_params["dropout_lstm"] = 0.0
        self.graph_params["dropout_hidden"] = 0.0

        # Updating of the default params if provided via flags as a dict
        self.graph_params = update_params(self.graph_params, update_fn_params, "Update_fn")

    def print_params(self):
        sorted_dict = sorted(self.graph_params.items(), key=lambda kv: kv[0])
        if len(sorted_dict) > 0:
            logging.info("update_fn_params:")
            for a in sorted_dict:
                logging.info(f"  {a[0]}: {a[1]}")

    def infer(self, inputs, is_training):
        with tf.compat.v1.variable_scope('update_function_LSTM'):
            h_dim = self.graph_params["hidden_node_feature_dim"]

            cellstate = inputs['cellstate']
            # cellstate: [num_nodes_in_batch, h_dim] float

            tensor_list = []

            # x: [num_nodes_in_batch, x_dim] float
            tensor_list += [inputs['x']]

            if self.graph_params["incorporate_hidden_features_in_update"]:
                # h: [num_nodes_in_batch, h_dim] float
                tensor_list += [inputs['h']]

            if self.graph_params["incorporate_node_input_features_in_update"]:
                if 'node_input_features_in_batch' in inputs:
                    # u: [num_nodes_in_batch, node_feature_dim] float
                    tensor_list += [inputs['node_input_features_in_batch']]
                else:
                    logging.warning("Node_input_features NOT in inputs. Could not use them.")

            # ===== calculate ingate-, outgate-, forgetgate-, and cellinput-activation
            ingate_activation = self._concat_combination(tensor_list, h_dim, is_training, activation=layers.sigmoid,
                                                         dropout_rate=self.graph_params["dropout_lstm"],
                                                         name='ingate_activation')
            outgate_activation = self._concat_combination(tensor_list, h_dim, is_training, activation=layers.sigmoid,
                                                          dropout_rate=self.graph_params["dropout_lstm"],
                                                          name='outgate_activation')
            forgetgate_activation = self._concat_combination(tensor_list, h_dim, is_training, activation=layers.sigmoid,
                                                             dropout_rate=self.graph_params["dropout_lstm"],
                                                             name='forgetgate_activation')
            cellinput_activation = self._concat_combination(tensor_list, h_dim, is_training, activation=layers.tanh,
                                                            dropout_rate=self.graph_params["dropout_lstm"],
                                                            name='cellinput_activation')
            # ingate_activation: [num_nodes_in_batch, h_dim] float
            # outgate_activation: [num_nodes_in_batch, h_dim] float
            # forgetgate_activation: [num_nodes_in_batch, h_dim] float
            # cellinput_activation: [num_nodes_in_batch, h_dim] float

            # ===== update hidden and cell states
            # cellstate: [num_nodes_in_batch, h_dim] float
            cellstate = forgetgate_activation * cellstate + ingate_activation * cellinput_activation
            # cellstate: [num_nodes_in_batch, h_dim] float
            h = outgate_activation * tf.tanh(cellstate)
            # h: [num_nodes_in_batch, h_dim] float
            if self.graph_params["dropout_hidden"] > 0.0:
                h = layers.dropout(h, is_training, self.graph_params["dropout_hidden"])

            output = dict()
            output['h'] = h
            output['cellstate'] = cellstate

            return output

    def get_hidden_dim(self):
        return self.graph_params["hidden_node_feature_dim"]

    def _concat_combination(self, tensor_list, output_dim, is_training, activation=None,
                            dropout_rate=0.0, name='concat_combination'):
        # tensor_list: list of [num_nodes_in_batch, old_dim] float  old_dim different for each tensor
        with tf.compat.v1.variable_scope(name):
            concatenated_tensor = tf.concat(tensor_list, axis=-1)
            # concatenated_tensor: [num_nodes_in_batch, sum_over old_dims]
            output = layers.ff_layer(concatenated_tensor, outD=output_dim, activation=activation, use_bias=True,
                                     is_training=is_training, reuse=tf.compat.v1.AUTO_REUSE)
            # output: [num_nodes_in_batch, output_dim] float
            if dropout_rate > 0.0:
                output = layers.dropout(output, is_training, dropout_rate)
            return output
