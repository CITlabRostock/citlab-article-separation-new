import tensorflow as tf
from collections import OrderedDict
from gnn.model.model_base import GraphBase
from gnn.model.graph_util import layers
from utils.flags import update_params


class ARU_v1_CNN(GraphBase):

    #   End point name  | End point subsampling name (scale_space_num: n)
    # ===================================================================
    # scale_0_unet_down_0_conv     | scale_0_unet_down_0_conv_subsamp
    # scale_0_unet_down_0_maxpool | scale_0_unet_down_0_max_pool_subsamp
    # scale_0_unet_down_1_conv     | scale_0_unet_down_1_conv_subsamp
    # scale_0_unet_down_1_maxpool | scale_0_unet_down_1_max_pool_subsamp
    # ...
    # scale_0_unet_down_n-2_conv     | scale_0_unet_down_n-2_conv_subsamp
    # scale_0_unet_down_n-2_maxpool | scale_0_unet_down_n-2_max_pool_subsamp
    # --------------------------------------------------------------------
    # scale_0_unet_down_n-1_conv     | scale_0_unet_down_n-1_conv_subsamp
    # --------------------------------------------------------------------
    # scale_0_unet_up_n-2_deconv     | scale_0_unet_up_n-2_deconv_subsamp
    # scale_0_unet_up_n-2_conv       | scale_0_unet_up_n-2_conv_subsamp
    # ...
    # scale_0_unet_up_1_deconv     | scale_0_unet_up_1_deconv_subsamp
    # scale_0_unet_up_1_conv       | scale_0_unet_up_1_conv_subsamp
    # scale_0_unet_up_0_deconv     | scale_0_unet_up_0_deconv_subsamp
    # scale_0_unet_up_0_conv       | scale_0_unet_up_0_conv_subsamp
    # --------------------------------------------------------------------
    # logits

    def __init__(self, params):
        super(ARU_v1_CNN, self).__init__(params)
        # Default configuration for the backbone graph
        self.graph_params["graph"] = 'RU'  # Which graph choose between: U, RU, ARU.
        self.graph_params["mvn"] = False  # Apply MVN for input images.
        self.graph_params["featRoot"] = 8  # Number of root features.
        self.graph_params["num_scales_att"] = 3  # Number of scale spaces for attention net.
        self.graph_params["scale_space_num"] = 5  # Number of scale spaces for (R)U - Net.
        self.graph_params["res_depth"] = 3  # Residual depth.
        self.graph_params["filter_size"] = 3  # Size of conv filter.
        self.graph_params["pool_size"] = 2  # Size of max pool filter.
        self.graph_params["activation_name"] = 'relu'  # Name of activation function, choose of: elu, relu, leaky

        # Updating of the default params if provided via flags as a dict
        self.graph_params = update_params(self.graph_params, self._flags.graph_backbone_params,
                                          "Backbone")

        # Define end_points similar to the Inception_v3 model
        self.end_points = {}

    def infer(self, inputs, is_training):
        """
        :param inputs: a tensor of size [batch_size, height, width, channels].
        :param is_training: Train mode or not
        :return:
        """
        logits, end_points = self._create_aru_net(inputs, is_training)

        return logits, end_points

    def _create_aru_net(self, inputs, is_training):
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

        with tf.compat.v1.variable_scope('aru_net') as scope:
            if self.graph_params["mvn"]:
                with tf.compat.v1.variable_scope('mvn') as scope:
                    inputs = tf.map_fn(lambda image: layers.per_image_standardization(image), inputs)

            with tf.compat.v1.variable_scope('misc') as scope:
                img_shape = tf.shape(inputs)
                # Shape of the upsampled tensor
                o_shape = tf.stack([
                    img_shape[0],
                    img_shape[1],
                    img_shape[2],
                    self.graph_params["featRoot"]
                ])

                useResidual = False
                useAttention = False

                if 'RU' in self.graph_params["graph"]:
                    useResidual = True
                if 'ARU' in self.graph_params["graph"]:
                    useAttention = True

                # Det Feature Maps
                out_det_map = OrderedDict()
                inp_scale_map = OrderedDict()
                inp_scale_map[0] = inputs
            if useAttention:
                with tf.compat.v1.variable_scope('attMapG') as scope:
                    for sc in range(1, self.graph_params["num_scales_att"]):
                        inp_scale_map[sc] = layers.avg_pool2d(inp_scale_map[sc - 1], ksize=[1, 2, 2, 1],
                                                              strides=[1, 2, 2, 1],
                                                              padding='SAME')
                    # Pay Attention
                    out_att_map = OrderedDict()
                    upSc = 8
                    for sc in range(0, self.graph_params["num_scales_att"]):
                        outAtt_O = self._attCNN(inp_scale_map[sc], activation, is_training=is_training)
                        outAtt_U = layers.upsample_simple(outAtt_O, tf.shape(inputs), upSc, 1)
                        scope.reuse_variables()
                        out_att_map[sc] = outAtt_U
                        upSc = upSc * 2
            with tf.compat.v1.variable_scope('featMapG') as scope:
                out_0 = self._detCNN(inputs, useResidual, False, self._flags.channels,
                                     self.graph_params["scale_space_num"], self.graph_params["res_depth"],
                                     self.graph_params["featRoot"], self.graph_params["filter_size"],
                                     self.graph_params["pool_size"], activation,
                                     is_training=is_training, sc=0)
                out_det_map[0] = out_0
                if useAttention:
                    scope.reuse_variables()
                    upSc = 1
                    for sc in range(1, self.graph_params["num_scales_att"]):
                        out_S = self._detCNN(inp_scale_map[sc], useResidual, False, self._flags.channels,
                                             self.graph_params["scale_space_num"],
                                             self.graph_params["res_depth"],
                                             self.graph_params["featRoot"], self.graph_params["filter_size"],
                                             self.graph_params["pool_size"], activation,
                                             is_training=is_training, sc=sc)
                        upSc = upSc * 2
                        out = layers.upsample_simple(out_S, o_shape, upSc, self.graph_params["featRoot"])
                        out_det_map[sc] = out

            with tf.compat.v1.variable_scope('logit') as scope:
                if useAttention:
                    val = []
                    for sc in range(0, self.graph_params["num_scales_att"]):
                        val.append(out_att_map[sc])
                    allAtt = tf.concat(values=val, axis=3)

                    # allAttSoftMax = layers.softmax(allAtt)
                    allAttSoftMax = layers.softmax(allAtt)
                    listOfAtt = tf.split(allAttSoftMax, self.graph_params["num_scales_att"], axis=3)
                    val = []
                    for sc in range(0, self.graph_params["num_scales_att"]):
                        val.append(tf.multiply(out_det_map[sc], listOfAtt[sc]))
                    map = tf.add_n(val)
                    self.end_points["sum_att_feat_map"] = map
                else:
                    map = out_det_map[0]

                logits = layers.conv2d(map, kernel_size=[4, 4], filters=self._flags.n_classes,
                                       activation=tf.identity, is_training=is_training, name='class')
                logits = tf.identity(logits, 'logits')

                self.end_points["logits"] = logits
        return self.end_points["logits"], self.end_points

    def _attCNN(self, input, activation, is_training):
        """
        Attention network
        :param input:
        :param activation:
        :return:
        """
        with tf.compat.v1.variable_scope('attPart') as scope:
            conv1 = layers.conv2d(input, kernel_size=[4, 4], filters=12, activation=activation,
                                  is_training=is_training, name='conv1')
            pool1 = layers.max_pool2d(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
            conv2 = layers.conv2d(pool1, kernel_size=[4, 4], filters=16, activation=activation,
                                  is_training=is_training, name='conv2')
            pool2 = layers.max_pool2d(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
            conv3 = layers.conv2d(pool2, kernel_size=[4, 4], filters=32, activation=activation,
                                  is_training=is_training, name='conv3')
            pool3 = layers.max_pool2d(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
            out_DS = layers.conv2d(pool3, kernel_size=[4, 4], filters=1, activation=activation,
                                   is_training=is_training, name='conv4')
        return out_DS

    def _detCNN(self, input, useResidual, useLSTM, channels, scale_space_num, res_depth, featRoot,
                filter_size, pool_size, activation, is_training, sc):
        """
        Feature Detection Network
        :param input:
        :param useResidual:
        :param useLSTM:
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
        for layer in range(0, scale_space_num):
            end_point = 'scale_' + str(sc) + '_unet_down_' + str(layer)
            with tf.compat.v1.variable_scope('unet_down_' + str(layer)) as scope:
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

                self.end_points[end_point + "_conv"] = dw_h_convs[layer]
                self.end_points[end_point + "_conv_subsamp"] = pool_size ** layer * sc

                if layer < scale_space_num - 1:
                    unetInp = layers.max_pool2d(dw_h_convs[layer], ksizePool, stridePool, padding='SAME', name='pool')
                    self.end_points[end_point + "_maxpool"] = unetInp
                    self.end_points[end_point + "_maxpool_subsamp"] = pool_size ** (layer + 1) * sc
                else:
                    unetInp = dw_h_convs[layer]
                lastFeatNum = actFeatNum
                actFeatNum *= pool_size

        actFeatNum = lastFeatNum // pool_size
        if useLSTM:
            # Run separable 2D LSTM
            unetInp = layers.separable_rnn(unetInp, lastFeatNum, scope="RNN2D", cellType='LSTM')
        for layer in range(scale_space_num - 2, -1, -1):
            end_point = 'scale_' + str(sc) + '_unet_up_' + str(layer)
            with tf.compat.v1.variable_scope('unet_up_' + str(layer)) as scope:
                # Upsampling followed by two ConvLayers
                dw_h_conv = dw_h_convs[layer]
                out_shape = tf.shape(dw_h_conv)
                deconv = layers.deconv2d(unetInp, kernel_shape=[filter_size, filter_size, actFeatNum, lastFeatNum],
                                         out_shape=out_shape, subS=pool_size, activation=activation,
                                         is_training=is_training, name='deconv')

                self.end_points[end_point + "_deconv"] = deconv
                self.end_points[end_point + "_deconv_subsamp"] = pool_size ** layer * sc

                conc = tf.concat([dw_h_conv, deconv], 3, name='concat')
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

                self.end_points[end_point + "_conv"] = unetInp
                self.end_points[end_point + "_conv_subsamp"] = pool_size ** layer * sc

        return unetInp
