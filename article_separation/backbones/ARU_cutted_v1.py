import tensorflow as tf
from gnn.model.model_base import GraphBase
from gnn.model.graph_util import layers
from utils.flags import update_params


class ARU_cutted_v1_CNN(GraphBase):
    def __init__(self, params):
        super(ARU_cutted_v1_CNN, self).__init__(params)
        # Default configuration for the backbone graph
        self.graph_params["mvn"] = True  # Apply MVN for input images.
        self.graph_params["featRoot"] = 12  # Number of root features.
        self.graph_params["num_scales_att"] = 3  # Number of scale spaces for attention net.
        self.graph_params["scale_space_num"] = 6  # Number of scale spaces for (R)U - Net.
        self.graph_params["res_depth"] = 0  # Residual depth.
        self.graph_params["filter_size"] = 3  # Size of conv filter.
        self.graph_params["pool_size"] = 2  # Size of max pool filter.
        self.graph_params["activation_name"] = 'relu'  # Name of activation function, choose of: elu, relu, leaky

        # Updating of the default params if provided via flags as a dict
        self.graph_params = update_params(self.graph_params, self._flags.graph_backbone_params,
                                          "Backbone")

    def infer(self, inputs, is_training):
        """
        :param inputs: a tensor of size [batch_size, height, width, channels].
        :param is_training: Train mode or not
        :return:
        """
        return self._create_cnn(inputs, is_training), {}

    def _create_cnn(self, input, is_training):
        activation = layers.relu
        if self.graph_params["activation_name"] == "elu":
            activation = layers.elu
        if self.graph_params["activation_name"] == "leaky":
            activation = layers.leaky_relu

        if self.graph_params["mvn"]:
            with tf.compat.v1.variable_scope('mvn') as scope:
                input = tf.map_fn(lambda image: layers.per_image_standardization(image), input)
        unetInp = input
        ksizePool = [1, self.graph_params["pool_size"], self.graph_params["pool_size"], 1]
        stridePool = ksizePool
        actFeatNum = self.graph_params["featRoot"]
        for layer in range(0, self.graph_params["scale_space_num"]):
            with tf.compat.v1.variable_scope('res_block_' + str(layer)) as scope:
                x = layers.conv2d(unetInp,
                                  kernel_size=[self.graph_params["filter_size"], self.graph_params["filter_size"]],
                                  filters=actFeatNum,
                                  activation=tf.identity, is_training=is_training, name='conv1')
                orig_x = x
                x = layers.relu(x, name='activation')
                if self.graph_params["res_depth"] > 0:
                    for aRes in range(0, self.graph_params["res_depth"]):
                        if aRes < self.graph_params["res_depth"] - 1:
                            x = layers.conv2d(x, kernel_size=[self.graph_params["filter_size"],
                                                              self.graph_params["filter_size"]], filters=actFeatNum,
                                              activation=activation,
                                              is_training=is_training, name='convR_' + str(aRes))
                        else:
                            x = layers.conv2d(x, kernel_size=[self.graph_params["filter_size"],
                                                              self.graph_params["filter_size"]], filters=actFeatNum,
                                              activation=tf.identity,
                                              is_training=is_training, name='convR_' + str(aRes))
                    x += orig_x
                    x = activation(x, name='activation')
                if layer < self.graph_params["scale_space_num"] - 1:
                    unetInp = layers.max_pool2d(x, ksizePool, stridePool, padding='SAME', name='pool')
                else:
                    unetInp = x
                actFeatNum *= self.graph_params["pool_size"]
        return unetInp
