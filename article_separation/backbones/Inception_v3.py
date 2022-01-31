import tensorflow as tf
from gnn.model.model_base import GraphBase
from gnn.model.graph_util import layers
from utils.flags import update_params


class Inception_v3_CNN(GraphBase):
    def __init__(self, params):
        super(Inception_v3_CNN, self).__init__(params)
        # Default configuration for the backbone graph
        self.graph_params["depth_multiplier"] = 1.0
        self.graph_params["min_depth"] = 16
        self.graph_params["start_point"] = 'Conv2d_1a_3x3'
        self.graph_params["end_point"] = 'Mixed_7c'
        # self.graph_params["mvn"] = False
        # self._graph_backbone_params["batch_norm"] = True

        # Updating of the default params if provided via flags as a dict
        self.graph_params = update_params(self.graph_params, self._flags.graph_backbone_params,
                                          "Backbone")

        if self.graph_params["depth_multiplier"] <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')

        # To make the model faster (by reduced feature depth) we introduce a depth function
        self._depth = lambda d: max(int(d * self.graph_params["depth_multiplier"]),
                                    self.graph_params["min_depth"])

    def infer(self, inputs, is_training):
        """Inception model from http://arxiv.org/abs/1512.00567.

        Constructs an Inception v3 network from inputs to the given final endpoint.
        This method can construct the network up to the final inception block
        Mixed_7c.

        Note that the names of the layers in the paper do not correspond to the names
        of the endpoints registered by this function although they build the same
        network.

        Here is a mapping from the old_names to the new names:
        Old name          | New name
        =======================================
        conv0             | Conv2d_1a_3x3
        conv1             | Conv2d_2a_3x3
        conv2             | Conv2d_2b_3x3
        pool1             | MaxPool_3a_3x3
        conv3             | Conv2d_3b_1x1
        conv4             | Conv2d_4a_3x3
        pool2             | MaxPool_5a_3x3
        mixed_35x35x256a  | Mixed_5b
        mixed_35x35x288a  | Mixed_5c
        mixed_35x35x288b  | Mixed_5d
        mixed_17x17x768a  | Mixed_6a
        mixed_17x17x768b  | Mixed_6b
        mixed_17x17x768c  | Mixed_6c
        mixed_17x17x768d  | Mixed_6d
        mixed_17x17x768e  | Mixed_6e
        mixed_8x8x1280a   | Mixed_7a
        mixed_8x8x2048a   | Mixed_7b
        mixed_8x8x2048b   | Mixed_7c

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          is_training: Train mode or not

        Returns:
          tensor_out: output tensor corresponding to the final_endpoint.
          end_points: a set of activations for external use, for example summaries or
                      losses.

        Raises:
          ValueError: if final_endpoint is not set to one of the predefined values,
                      or depth_multiplier <= 0
        """
        # end_points will collect relevant activations for external use, for example
        # summaries or losses.
        end_points = {}

        initial_startpoint = self.graph_params["start_point"]
        final_endpoint = self.graph_params["end_point"]

        with tf.compat.v1.variable_scope('InceptionV3'):
            # This bool indicates whether out starting point was already reached
            started = False
            net = inputs

            # 299 x 299 x 3
            end_point = 'Conv2d_1a_3x3'

            if initial_startpoint == end_point or started:
                started = True
                net = layers.conv2d(net, kernel_size=[3, 3], filters=self._depth(32),
                                    is_training=is_training, strides=[1, 2, 2, 1],
                                    padding='VALID', name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # 149 x 149 x 32
            end_point = 'Conv2d_2a_3x3'
            if initial_startpoint == end_point or started:
                started = True
                net = layers.conv2d(net, kernel_size=[3, 3], filters=self._depth(32),
                                    is_training=is_training, strides=[1, 1, 1, 1],
                                    padding='VALID', name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # 147 x 147 x 32
            end_point = 'Conv2d_2b_3x3'
            if initial_startpoint == end_point or started:
                started = True
                net = layers.conv2d(net, kernel_size=[3, 3], filters=self._depth(64),
                                    is_training=is_training, strides=[1, 1, 1, 1],
                                    padding='SAME', name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # 147 x 147 x 64
            end_point = 'MaxPool_3a_3x3'
            if initial_startpoint == end_point or started:
                started = True
                net = layers.max_pool2d(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # 73 x 73 x 64
            end_point = 'Conv2d_3b_1x1'
            if initial_startpoint == end_point or started:
                started = True
                net = layers.conv2d(net, kernel_size=[1, 1], filters=self._depth(80),
                                    is_training=is_training, strides=[1, 1, 1, 1],
                                    padding='VALID', name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # 73 x 73 x 80.
            end_point = 'Conv2d_4a_3x3'
            if initial_startpoint == end_point or started:
                started = True
                net = layers.conv2d(net, kernel_size=[3, 3], filters=self._depth(192),
                                    is_training=is_training, strides=[1, 1, 1, 1],
                                    padding='VALID', name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # 71 x 71 x 192.
            end_point = 'MaxPool_5a_3x3'
            if initial_startpoint == end_point or started:
                started = True
                net = layers.max_pool2d(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # 35 x 35 x 192.

            # Inception blocks
            # mixed: 35 x 35 x 288.
            end_point = 'Mixed_5b'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_a(net, is_training=is_training, scope_name=end_point,
                                        branch_3_depth=self._depth(32))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed_1: 35 x 35 x 288.
            end_point = 'Mixed_5c'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_a(net, is_training=is_training, scope_name=end_point,
                                        branch_3_depth=self._depth(64))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed_2: 35 x 35 x 288.
            end_point = 'Mixed_5d'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_a(net, is_training=is_training, scope_name=end_point,
                                        branch_3_depth=self._depth(64))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed_3: 17 x 17 x 768.
            end_point = 'Mixed_6a'
            if initial_startpoint == end_point or started:
                started = True
                with tf.compat.v1.variable_scope(end_point):
                    with tf.compat.v1.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(net, kernel_size=[3, 3], filters=self._depth(384),
                                                 is_training=is_training, strides=[1, 2, 2, 1],
                                                 padding='VALID', name="Conv2d_1a_1x1")
                    with tf.compat.v1.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(net, kernel_size=[1, 1], filters=self._depth(64),
                                                 is_training=is_training, strides=[1, 1, 1, 1],
                                                 padding='SAME', name="Conv2d_0a_1x1")
                        branch_1 = layers.conv2d(branch_1, kernel_size=[3, 3], filters=self._depth(96),
                                                 is_training=is_training, strides=[1, 1, 1, 1],
                                                 padding='SAME', name="Conv2d_0b_3x3")
                        branch_1 = layers.conv2d(branch_1, kernel_size=[3, 3], filters=self._depth(96),
                                                 is_training=is_training, strides=[1, 2, 2, 1],
                                                 padding='VALID', name="Conv2d_1a_1x1")
                    with tf.compat.v1.variable_scope('Branch_2'):
                        branch_2 = layers.max_pool2d(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                                                     name='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed4: 17 x 17 x 768.
            end_point = 'Mixed_6b'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_b(net, is_training=is_training, scope_name=end_point,
                                        hidden_depth=self._depth(128))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed_5: 17 x 17 x 768.
            end_point = 'Mixed_6c'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_b(net, is_training=is_training, scope_name=end_point,
                                        hidden_depth=self._depth(160))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # mixed_6: 17 x 17 x 768.
            end_point = 'Mixed_6d'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_b(net, is_training=is_training, scope_name=end_point,
                                        hidden_depth=self._depth(160))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed_7: 17 x 17 x 768.
            end_point = 'Mixed_6e'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_b(net, is_training=is_training, scope_name=end_point,
                                        hidden_depth=self._depth(192))
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed_8: 8 x 8 x 1280.
            end_point = 'Mixed_7a'
            if initial_startpoint == end_point or started:
                started = True
                with tf.compat.v1.variable_scope(end_point):
                    with tf.compat.v1.variable_scope('Branch_0'):
                        branch_0 = layers.conv2d(net,
                                                 kernel_size=[1, 1], filters=self._depth(192),
                                                 is_training=is_training, strides=[1, 1, 1, 1],
                                                 padding='SAME', name="Conv2d_0a_1x1")
                        branch_0 = layers.conv2d(branch_0,
                                                 kernel_size=[3, 3],
                                                 filters=self._depth(320),
                                                 is_training=is_training, strides=[1, 2, 2, 1],
                                                 padding='VALID', name="Conv2d_1a_3x3")
                    with tf.compat.v1.variable_scope('Branch_1'):
                        branch_1 = layers.conv2d(net,
                                                 kernel_size=[1, 1], filters=self._depth(192),
                                                 is_training=is_training, strides=[1, 1, 1, 1],
                                                 padding='SAME', name="Conv2d_0a_1x1")
                        branch_1 = layers.conv2d(branch_1,
                                                 kernel_size=[1, 7],
                                                 filters=self._depth(192),
                                                 is_training=is_training, strides=[1, 1, 1, 1],
                                                 padding='SAME', name="Conv2d_0b_1x7")
                        branch_1 = layers.conv2d(branch_1,
                                                 kernel_size=[7, 1],
                                                 filters=self._depth(192),
                                                 is_training=is_training, strides=[1, 1, 1, 1],
                                                 padding='SAME', name="Conv2d_0c_7x1")
                        branch_1 = layers.conv2d(branch_1,
                                                 kernel_size=[3, 3],
                                                 filters=self._depth(192),
                                                 is_training=is_training, strides=[1, 2, 2, 1],
                                                 padding='VALID', name="Conv2d_1a_3x3")
                    with tf.compat.v1.variable_scope('Branch_2'):
                        branch_2 = layers.max_pool2d(net, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID',
                                                     name='MaxPool_1a_3x3')
                    net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            # mixed_9: 8 x 8 x 2048.
            end_point = 'Mixed_7b'
            if initial_startpoint == end_point or started:
                started = True
                net = self._inc_block_c(net, is_training=is_training, scope_name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points

            # mixed_10: 8 x 8 x 2048.
            end_point = 'Mixed_7c'
            if initial_startpoint == end_point or started:
                net = self._inc_block_c(net, is_training=is_training, scope_name=end_point)
                end_points[end_point] = net
                if end_point == final_endpoint:
                    return net, end_points
            raise ValueError(f'Unknown start or final endpoint {final_endpoint}')

    def _inc_block_a(self, net, is_training, scope_name, branch_3_depth):
        with tf.compat.v1.variable_scope(scope_name):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = layers.conv2d(net,
                                         kernel_size=[1, 1], filters=self._depth(64),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0a_1x1')
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = layers.conv2d(net,
                                         kernel_size=[1, 1], filters=self._depth(48),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = layers.conv2d(branch_1,
                                         kernel_size=[5, 5], filters=self._depth(64),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0b_5x5')
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = layers.conv2d(net,
                                         kernel_size=[1, 1], filters=self._depth(64),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = layers.conv2d(branch_2,
                                         kernel_size=[3, 3], filters=self._depth(96),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0b_3x3')
                branch_2 = layers.conv2d(branch_2,
                                         kernel_size=[3, 3], filters=self._depth(96),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0c_3x3')
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = layers.avg_pool2d(net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                             name='AvgPool_0a_3x3')
                branch_3 = layers.conv2d(branch_3,
                                         kernel_size=[1, 1], filters=branch_3_depth,
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    def _inc_block_b(self, net, is_training, scope_name, hidden_depth):
        with tf.compat.v1.variable_scope(scope_name):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = layers.conv2d(net,
                                         kernel_size=[1, 1], filters=self._depth(192),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0a_1x1')
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = layers.conv2d(net,
                                         kernel_size=[1, 1], filters=hidden_depth,
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0a_1x1')
                branch_1 = layers.conv2d(branch_1,
                                         kernel_size=[1, 7], filters=hidden_depth,
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0b_1x7')
                branch_1 = layers.conv2d(branch_1,
                                         kernel_size=[7, 1], filters=self._depth(192),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0c_7x1')
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = layers.conv2d(net,
                                         kernel_size=[1, 1], filters=hidden_depth,
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0a_1x1')
                branch_2 = layers.conv2d(branch_2,
                                         kernel_size=[7, 1], filters=hidden_depth,
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0b_7x1')
                branch_2 = layers.conv2d(branch_2,
                                         kernel_size=[1, 7], filters=hidden_depth,
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0c_1x7')
                branch_2 = layers.conv2d(branch_2,
                                         kernel_size=[7, 1], filters=hidden_depth,
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0d_7x1')
                branch_2 = layers.conv2d(branch_2,
                                         kernel_size=[1, 7], filters=self._depth(192),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0e_1x7')
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = layers.avg_pool2d(net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                             name='AvgPool_0a_3x3')
                branch_3 = layers.conv2d(branch_3, kernel_size=[1, 1], filters=self._depth(192),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name='Conv2d_0b_1x1')
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    def _inc_block_c(self, net, is_training, scope_name):
        with tf.compat.v1.variable_scope(scope_name):
            with tf.compat.v1.variable_scope('Branch_0'):
                branch_0 = layers.conv2d(net, kernel_size=[1, 1], filters=self._depth(320),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name="Conv2d_0a_1x1")
            with tf.compat.v1.variable_scope('Branch_1'):
                branch_1 = layers.conv2d(net, kernel_size=[1, 1], filters=self._depth(384),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name="Conv2d_0a_1x1")
                branch_1a = layers.conv2d(branch_1, kernel_size=[1, 3], filters=self._depth(384),
                                          is_training=is_training, strides=[1, 1, 1, 1],
                                          padding='SAME', name="Conv2d_0b_1x3")
                branch_1b = layers.conv2d(branch_1, kernel_size=[3, 1], filters=self._depth(384),
                                          is_training=is_training, strides=[1, 1, 1, 1],
                                          padding='SAME', name="Conv2d_0c_3x1")

                branch_1 = tf.concat(axis=3, values=[branch_1a, branch_1b])
            with tf.compat.v1.variable_scope('Branch_2'):
                branch_2 = layers.conv2d(net, kernel_size=[1, 1], filters=self._depth(448),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name="Conv2d_0a_1x1")
                branch_2 = layers.conv2d(branch_2, kernel_size=[3, 3], filters=self._depth(384),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name="Conv2d_0b_3x3")
                branch_2a = layers.conv2d(branch_2, kernel_size=[1, 3], filters=self._depth(384),
                                          is_training=is_training, strides=[1, 1, 1, 1],
                                          padding='SAME', name="Conv2d_0c_1x3")
                branch_2b = layers.conv2d(branch_2, kernel_size=[3, 1], filters=self._depth(384),
                                          is_training=is_training, strides=[1, 1, 1, 1],
                                          padding='SAME', name="Conv2d_0d_3x1")
                branch_2 = tf.concat(axis=3, values=[branch_2a, branch_2b])
            with tf.compat.v1.variable_scope('Branch_3'):
                branch_3 = layers.avg_pool2d(net, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                             name='AvgPool_0a_3x3')
                branch_3 = layers.conv2d(branch_3, kernel_size=[1, 1], filters=self._depth(192),
                                         is_training=is_training, strides=[1, 1, 1, 1],
                                         padding='SAME', name="Conv2d_0b_1x1")
            return tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])

    # <editor-fold desc="Full Classification Stuff, has to be checked.">
    def infer_full(self, inputs,
                   num_classes=1000,
                   is_training=True,
                   dropout_keep_prob=0.8,
                   prediction_fn=layers.softmax,
                   spatial_squeeze=True,
                   reuse=None,
                   create_aux_logits=True,
                   scope='InceptionV3',
                   global_pool=False):
        """Inception model from http://arxiv.org/abs/1512.00567.

        "Rethinking the Inception Architecture for Computer Vision"

        Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
        Zbigniew Wojna.

        With the default arguments this method constructs the exact model defined in
        the paper. However, one can experiment with variations of the inception_v3
        network by changing arguments dropout_keep_prob, min_depth and
        depth_multiplier.

        The default image size used to train this network is 299x299.

        Args:
          inputs: a tensor of size [batch_size, height, width, channels].
          num_classes: number of predicted classes. If 0 or None, the logits layer
            is omitted and the input features to the logits layer (before dropout)
            are returned instead.
          is_training: whether is training or not.
          dropout_keep_prob: the percentage of activation values that are retained.
          prediction_fn: a function to get predictions out of logits.
          spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
              shape [B, 1, 1, C], where B is batch_size and C is number of classes.
          reuse: whether or not the network and its variables should be reused. To be
            able to reuse 'scope' must be given.
          create_aux_logits: Whether to create the auxiliary logits.
          scope: Optional variable_scope.
          global_pool: Optional boolean flag to control the avgpooling before the
            logits layer. If false or unset, pooling is done with a fixed window
            that reduces default-sized inputs to 1x1, while larger inputs lead to
            larger outputs. If true, any input size is pooled down to 1x1.

        Returns:
          net: a Tensor with the logits (pre-softmax activations) if num_classes
            is a non-zero integer, or the non-dropped-out input to the logits layer
            if num_classes is 0 or None.
          end_points: a dictionary from components of the network to the corresponding
            activation.

        Raises:
          ValueError: if 'depth_multiplier' is less than or equal to zero.
        """

        with tf.compat.v1.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse):
            net, end_points = self.infer(inputs, is_training=is_training)

            # Auxiliary Head logits
            if create_aux_logits and num_classes:
                aux_logits = end_points['Mixed_6e']
                with tf.compat.v1.variable_scope('AuxLogits'):
                    aux_logits = layers.avg_pool2d(aux_logits, ksize=[1, 5, 5, 1], strides=[1, 3, 3, 1],
                                                   padding='VALID',
                                                   name='AvgPool_1a_5x5')
                    aux_logits = layers.conv2d(aux_logits,
                                               kernel_size=[1, 1],
                                               filters=self._depth(128),
                                               is_training=is_training, strides=[1, 1, 1, 1],
                                               padding='SAME', name="Conv2d_1b_1x1")
                    # Shape of feature map before the final layer.
                    kernel_size = self._reduced_kernel_size_for_small_input(aux_logits, [5, 5])
                    aux_logits = layers.conv2d(aux_logits,
                                               kernel_size=[1, 1],
                                               filters=self._depth(768),
                                               is_training=is_training, strides=[1, 1, 1, 1],
                                               padding='VALID', initOpt=-0.01,
                                               name=f'Conv2d_2a_{kernel_size[0]}x{kernel_size[1]}')
                    aux_logits = layers.conv2d(aux_logits,
                                               kernel_size=[1, 1], filters=num_classes,
                                               is_training=is_training, strides=[1, 1, 1, 1],
                                               padding='VALID', initOpt=-0.001, name='Conv2d_2b_1x1')
                    if spatial_squeeze:
                        aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
                    end_points['AuxLogits'] = aux_logits

            # Final pooling and prediction
            with tf.compat.v1.variable_scope('Logits'):
                if global_pool:
                    # Global average pooling.
                    net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')
                    end_points['global_pool'] = net
                else:
                    # Pooling with a fixed kernel size.
                    kernel_size = self._reduced_kernel_size_for_small_input(net, [8, 8])
                    net = layers.avg_pool2d(net, ksize=[1, kernel_size[0], kernel_size[1], 1], strides=[1, 1, 1, 1],
                                            padding='VALID',
                                            name=f'AvgPool_1a_{kernel_size[0]}x{kernel_size[1]}')
                    end_points['AvgPool_1a'] = net
                if not num_classes:
                    return net, end_points
                # 1 x 1 x 2048
                net = layers.dropout(net, is_training=is_training, keep_prob=dropout_keep_prob, name='Dropout_1b')
                end_points['PreLogits'] = net
                # 2048
                logits = layers.conv2d(net,
                                       kernel_size=[1, 1], filters=num_classes,
                                       is_training=is_training, strides=[1, 1, 1, 1],
                                       padding='SAME', activation=None, name="Conv2d_1c_1x1")
                if spatial_squeeze:
                    logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
                # 1000
            end_points['Logits'] = logits
            end_points['Predictions'] = prediction_fn(logits, name='Predictions')
        return logits, end_points

    def _reduced_kernel_size_for_small_input(self, input_tensor, kernel_size):
        """Define kernel size which is automatically reduced for small input.

        If the shape of the input images is unknown at graph construction time this
        function assumes that the input images are is large enough.

        Args:
          input_tensor: input tensor of size [batch_size, height, width, channels].
          kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

        Returns:
          a tensor with the kernel size.

        TODO(jrru): Make this function work with unknown shapes. Theoretically, this
        can be done with the code below. Problems are two-fold: (1) If the shape was
        known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
        handle tensors that define the kernel size.
            shape = tf.shape(input_tensor)
            return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                               tf.minimum(shape[2], kernel_size[1])])

        """
        shape = input_tensor.get_shape().as_list()
        if shape[1] is None or shape[2] is None:
            kernel_size_out = kernel_size
        else:
            kernel_size_out = [min(shape[1], kernel_size[0]),
                               min(shape[2], kernel_size[1])]
        return kernel_size_out
    # </editor-fold>
