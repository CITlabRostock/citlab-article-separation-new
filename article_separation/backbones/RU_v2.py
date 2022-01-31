import tensorflow as tf
from collections import OrderedDict
from gnn.model.model_base import GraphBase
from gnn.model.graph_util import layers
from utils.flags import update_params


class RU_v2_CNN(GraphBase):
    def __init__(self, params):
        super(RU_v2_CNN, self).__init__(params)
        # Default configuration for the backbone graph
        self.graph_params["mvn"] = False  # Apply MVN for input images.
        self.graph_params["inp4up"] = True  # Concat input in upsample path
        self.graph_params["featRoot"] = 8  # Number of root features.
        self.graph_params["scale_space_num"] = 5  # Number of scale spaces for RU - Net.
        self.graph_params["res_depth"] = 3  # Residual depth.
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
        return self._create_ru_net(inputs, is_training), {}

    def _create_ru_net(self, inputs, is_training):
        """
        Creates a neural pixel labeler of specified type. This NPL can process images of arbitrarily sizes

        :param inputs: a tensor of size [batch_size, height, width, channels].
        :param is_training: self explanatory
        """

        # What activation function?
        activation = layers.relu
        if self.graph_params["activation_name"] == "elu":
            activation = layers.elu
        if self.graph_params["activation_name"] == "leaky":
            activation = layers.leaky_relu

        with tf.compat.v1.variable_scope('ru_net') as scope:
            if self.graph_params["mvn"]:
                with tf.compat.v1.variable_scope('mvn') as scope:
                    inputs = tf.map_fn(lambda image: layers.per_image_standardization(image), inputs)
            with tf.compat.v1.variable_scope('misc') as scope:
                # Det Feature Maps
                out_det_map = OrderedDict()
            with tf.compat.v1.variable_scope('featMapG') as scope:
                out_0 = self._detCNN(inputs, True, self._flags.channels, self.graph_params["scale_space_num"],
                                     self.graph_params["res_depth"],
                                     self.graph_params["featRoot"], self.graph_params["filter_size"],
                                     self.graph_params["pool_size"], activation,
                                     is_training=is_training)
                out_det_map[0] = out_0

            with tf.compat.v1.variable_scope('logit') as scope:
                map = out_det_map[0]
                logits = layers.conv2d(map, kernel_size=[4, 4], filters=self._flags.n_classes,
                                       activation=tf.identity, is_training=is_training, name='class')
                logits = tf.identity(logits, 'logits')
        return logits

    def _detCNN(self, input, useResidual, channels, scale_space_num, res_depth, featRoot,
                filter_size, pool_size, activation, is_training):
        """
        Feature Detection Network
        :param input:
        :param useResidual:
        :param channels:
        :param scale_space_num:
        :param res_depth:
        :param featRoot:
        :param filter_size:
        :param pool_size:
        :param activation:
        :return:
        """
        unetInp = input
        ksizePool = [1, pool_size, pool_size, 1]
        stridePool = ksizePool
        lastFeatNum = channels
        actFeatNum = featRoot
        dw_h_convs = OrderedDict()

        inp_scale_map = OrderedDict()
        inp_scale_map[0] = unetInp
        for sc in range(1, scale_space_num):
            inp_scale_map[sc] = layers.avg_pool2d(inp_scale_map[sc - 1], ksizePool, stridePool,
                                                  padding='SAME')

        for layer in range(0, scale_space_num):
            with tf.compat.v1.variable_scope('unet_down_' + str(layer)) as scope:
                if layer > 0:
                    unetInp = tf.concat([unetInp, inp_scale_map[layer]], axis=3)
                if useResidual:
                    x = layers.conv2d(unetInp, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                      activation=tf.identity, is_training=is_training, name='conv1')
                    orig_x = x
                    x = layers.relu(x, name='activation')
                    for aRes in range(0, res_depth):
                        if aRes < res_depth - 1:
                            x = layers.conv2d(x, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                              activation=activation,
                                              is_training=is_training, name='convR_' + str(aRes))
                        else:
                            x = layers.conv2d(x, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                              activation=tf.identity,
                                              is_training=is_training, name='convR_' + str(aRes))
                    x += orig_x
                    x = activation(x, name='activation')
                    dw_h_convs[layer] = x
                else:
                    conv1 = layers.conv2d(unetInp, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                          activation=activation, is_training=is_training, name='conv1')
                    dw_h_convs[layer] = layers.conv2d(conv1, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                                      activation=activation,
                                                      is_training=is_training, name='conv2')
                if layer < scale_space_num - 1:
                    unetInp = layers.max_pool2d(dw_h_convs[layer], ksizePool, stridePool, padding='SAME', name='pool')
                else:
                    unetInp = dw_h_convs[layer]
                lastFeatNum = actFeatNum
                actFeatNum *= pool_size
        actFeatNum = lastFeatNum // pool_size
        for layer in range(scale_space_num - 2, -1, -1):
            with tf.compat.v1.variable_scope('unet_up_' + str(layer)) as scope:
                # Upsampling followed by two ConvLayers
                dw_h_conv = dw_h_convs[layer]
                out_shape = tf.shape(dw_h_conv)

                deconv = layers.deconv2d(unetInp, kernel_shape=[filter_size, filter_size, actFeatNum, lastFeatNum],
                                         out_shape=out_shape, subS=pool_size, activation=activation,
                                         is_training=is_training, name='deconv')

                conc = tf.concat([dw_h_conv, deconv], 3, name='concat')
                if self.graph_params["inp4up"]:
                    conc = tf.concat([conc, inp_scale_map[layer]], axis=3)
                if useResidual:
                    x = layers.conv2d(conc, kernel_size=[filter_size, filter_size],
                                      filters=actFeatNum, activation=tf.identity,
                                      is_training=is_training, name='conv1')
                    orig_x = x
                    x = layers.relu(x, name='activation')
                    for aRes in range(0, res_depth):
                        if aRes < res_depth - 1:
                            x = layers.conv2d(x, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                              activation=activation,
                                              is_training=is_training, name='convR_' + str(aRes))
                        else:
                            x = layers.conv2d(x, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                              activation=tf.identity,
                                              is_training=is_training, name='convR_' + str(aRes))
                    x += orig_x
                    unetInp = activation(x, name='activation')
                else:
                    conv1 = layers.conv2d(conc, kernel_size=[filter_size, filter_size],
                                          filters=actFeatNum, activation=activation,
                                          is_training=is_training, name='conv1')
                    unetInp = layers.conv2d(conv1, kernel_size=[filter_size, filter_size], filters=actFeatNum,
                                            activation=activation, is_training=is_training, name='conv2')
                lastFeatNum = actFeatNum
                actFeatNum /= pool_size
        return unetInp
