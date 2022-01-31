import numpy as np
import tensorflow as tf
from tensorflow.contrib.cudnn_rnn import CudnnCompatibleLSTMCell, CudnnCompatibleGRUCell
from tensorflow.contrib.cudnn_rnn import CudnnLSTM, CudnnGRU
from tensorflow.contrib.layers import batch_norm as batch_norm_tf
from tensorflow.python.util.tf_export import tf_export


# <editor-fold desc="Activation Functions (leaky_relu, relu, elu, sigmoid, tanh, softmax)">
def leaky_relu(x, leak=0.1, name="leakyRelu"):
    """
    Leaky ReLU activation function as proposed in the paper:
    http://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf

    It allows a small, non-zero gradient when the unit is not active.

    Args:
        x: input tensor.
        leak: `float`, hyperparameter that controls the slope of the negative part. It corresponds
            to `1/a` in the original paper and should take on small positive values between 0 and 1.
            If it's 0, the Leaky ReLU becomes the standard ReLU.
        name: `str`, the name scope.

    Returns:
        output tensor with the same shape as `x`.
    """
    assert 0.0 <= leak <= 1.0
    with tf.compat.v1.variable_scope(name):
        return tf.maximum(0.0, x) + tf.multiply(leak, tf.minimum(0.0, x))


def relu(features, name=None):
    return tf.nn.relu(features, name=name)


def elu(features, name=None):
    return tf.nn.elu(features, name=name)


def sigmoid(x, name=None):
    return tf.nn.sigmoid(x, name=name)


def tanh(x, name=None):
    return tf.nn.tanh(x, name=name)


def softmax(inputs, axis=None, name=None):
    return tf.nn.softmax(inputs, axis=axis, name=name)


# </editor-fold>

# <editor-fold desc="Trainable Layers (Feedforward Layer, Recurrent Layer)">

# <editor-fold desc="Feedforward Layer (ff_layer, conv1d, conv2d, sep_conv2d, dil_conv2d, deconv2d)">
def ff_layer(inputs,
             outD,
             is_training,
             activation=relu,
             use_bias=True,
             use_bn=False,
             initOpt=0,
             biasInit=0.1,
             name="dense",
             reuse=None):
    """
    FeedForward layer supports processing of entire feature maps (positional wise classification converning the last dimensions features).
    :param inputs:
    :param outD: `int`, number of output features.
    :param is_training:
    :param activation:
    :param use_bias:
    :param use_bn:
    :param biasInit:
    :param name:
    :return:
    """
    with tf.compat.v1.variable_scope(name, reuse=reuse):
        stddev = 5e-2
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (inputs.get_shape().as_list()[-1] + outD))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (inputs.shape[-1])), 5e-2)
        initializer = tf.random_normal_initializer(stddev=stddev)
        if initOpt < 0:
            initializer = tf.random.truncated_normal_initializer(0.0, -initOpt)

        W = tf.compat.v1.get_variable("weights", [inputs.shape[-1], outD],
                                      initializer=initializer)
        if len(inputs.shape) > 2:
            # Broadcasting is required for the inputs if rank is greater than 2.
            outputs = tf.tensordot(inputs, W, [[len(inputs.shape) - 1], [0]], name="Wx")
            # Reshape the output back to the original ndim of the input.
            # shape = tf.shape(inputs)
            shape = inputs.shape.as_list()
            output_shape = shape[:-1] + [outD]
            outputs.set_shape(output_shape)
        else:
            outputs = tf.matmul(inputs, W, name='Wx')
        if use_bias:
            bias = tf.compat.v1.get_variable("bias", outD,
                                             initializer=tf.constant_initializer(value=biasInit))
            outputs = tf.nn.bias_add(outputs, bias, name='preActivation')
        if use_bn:
            outputs = batch_norm(outputs, is_training, scope="batchNorm")
        if activation:
            outputs = activation(outputs, name='activation')
        return outputs


def batch_norm(inputs, is_training, scope="batchNorm"):
    return batch_norm_tf(inputs, is_training=is_training, scale=True, fused=True, scope=scope)


def layer_norm(inputs, eps=0.00001, elementwise_affine=True, name="layerNorm", reuse=None):
    """Applies layer normalization to a tensor given `inputs`. The normalization is performed for the last dimension of this tensor

    Returns:
        tensor, which has the same type and shape as `inputs`
    """

    with tf.compat.v1.variable_scope(name, reuse=reuse):
        mean, var = tf.nn.moments(inputs, axes=[-1])
        mean = tf.expand_dims(mean, axis=-1)
        var = tf.expand_dims(var, axis=-1)
        x = (inputs - mean) / tf.sqrt(var + eps)
        if elementwise_affine:
            gamma = tf.compat.v1.get_variable("gamma", [inputs.get_shape().as_list()[-1]],
                                              initializer=tf.constant_initializer(value=1.0))
            beta = tf.compat.v1.get_variable("beta", [inputs.get_shape().as_list()[-1]],
                                             initializer=tf.constant_initializer(value=0.0))
        else:
            gamma = tf.compat.v1.get_variable("gamma", 1, initializer=tf.constant_initializer(value=1.0))
            beta = tf.compat.v1.get_variable("beta", 1, initializer=tf.constant_initializer(value=0.0))
        x = gamma * x + beta
        return x


def conv1d(inputs, is_training,  # to be used later ?!
           kernel_width,
           filters,
           stride=1,
           activation=relu,
           padding='SAME',
           use_bias=True,
           initOpt=0,
           biasInit=0.1,
           drop_rate=0.0,
           name='conv1d'):
    """Adds a 1-D convolutional layer given 3-D `inputs`.


    Returns:
        `3-D Tensor`, has the same type `inputs`.
    """
    with tf.compat.v1.variable_scope(name):
        kernel_shape = [kernel_width, inputs.get_shape().as_list()[-1], filters]  # [width, inFeat, outFeat]
        strides = [1, 1, stride, 1]
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] + kernel_shape[2]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1])), 5e-2)
        initializer = tf.random_normal_initializer(stddev=stddev)
        if initOpt < 0:
            initializer = tf.random.truncated_normal_initializer(0.0, -initOpt)

        kernel_shape = [1, kernel_shape[0], kernel_shape[1], kernel_shape[2]]
        inputs = tf.expand_dims(inputs, axis=1)

        kernel = tf.compat.v1.get_variable("weights", kernel_shape,
                                           initializer=initializer)
        # Warum hier conv2d???
        outputs = conv2d_op(inputs, kernel, strides, padding=padding, name='conv')
        if use_bias:
            bias = tf.compat.v1.get_variable("biases", kernel_shape[3],
                                             initializer=tf.constant_initializer(value=biasInit))
            outputs = tf.nn.bias_add(outputs, bias, name='preActivation')
        if activation:
            outputs = activation(outputs, name='activation')
        outputs = tf.squeeze(outputs, axis=1)
        if drop_rate > 0.0:
            outputs = dropout(outputs, is_training=is_training, rate=drop_rate)
        return outputs


def conv2d(inputs, is_training,  # to be used later ?!
           kernel_size,
           filters,
           strides=None,
           activation=relu,
           padding='SAME',
           use_bias=True,
           initOpt=0,
           biasInit=0.1,
           drop_rate=0.0,
           batchNorm=False,
           name="conv2d"):
    """Adds a 2-D convolutional layer given 4-D `inputs` and `kernel`.

    Args:
        scope_or_name: `string` or `VariableScope`, the scope to open.
        inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
        is_training: `bool`, whether or not the layer is in training mode.
        kernel_size: list of `ints`, length 2, [kernel_height, kernel_width].
        filters: `int`, number of output filter.
        strides: list of `ints`, length 4, the stride of the sliding window for each dimension of `inputs`.
        activation: activation function to be used (default: `relu`).

        padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.

    Returns:
        `4-D Tensor`, has the same type `inputs`.
    """
    with tf.compat.v1.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape().as_list()[-1], filters]
        if strides is None:
            strides = [1, 1, 1, 1]
        stddev = 5e-2
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])), 5e-2)
        initializer = tf.random_normal_initializer(stddev=stddev)
        if initOpt < 0:
            initializer = tf.random.truncated_normal_initializer(0.0, -initOpt)

        kernel = tf.compat.v1.get_variable("weights", kernel_shape,
                                           initializer=initializer)
        outputs = conv2d_op(inputs, kernel, strides, padding=padding, name='conv')
        if use_bias:
            bias = tf.compat.v1.get_variable("biases", kernel_shape[3],
                                             initializer=tf.constant_initializer(value=biasInit))
            outputs = tf.nn.bias_add(outputs, bias, name='preActivation')
        if batchNorm:
            outputs = tf.compat.v1.layers.batch_normalization(outputs, training=is_training)
        if activation:
            outputs = activation(outputs, name='activation')
        if drop_rate > 0.0:
            outputs = dropout(outputs, is_training=is_training, rate=drop_rate)
        return outputs


def sep_conv2d(inputs,
               is_training,
               kernel_size,
               filters,
               depth_multiplier,
               strides=None,
               activation=relu,
               drop_rate=0.0,
               initOpt=0,
               biasInit=0.1,
               padding='SAME',
               name='sep_conv2d'):
    with tf.compat.v1.variable_scope(name):
        kernel_shape = [kernel_size[0], kernel_size[1], inputs.get_shape().as_list()[-1], filters]
        if strides is None:
            strides = [1, 1, 1, 1]
        if initOpt == 0:
            stddev1 = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + 1))
            stddev2 = np.sqrt(2.0 / (kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev1 = 5e-2
            stddev2 = 5e-2
        if initOpt == 2:
            stddev1 = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])), 5e-2)
            stddev2 = min(np.sqrt(2.0 / (kernel_shape[2])), 5e-2)
        kernel1 = tf.compat.v1.get_variable("weights_sep",
                                            [kernel_shape[0], kernel_shape[1], kernel_shape[2], depth_multiplier],
                                            initializer=tf.random_normal_initializer(stddev=stddev1))
        kernel2 = tf.compat.v1.get_variable("weights_1x1", [1, 1, depth_multiplier * kernel_shape[2], kernel_shape[3]],
                                            initializer=tf.random_normal_initializer(stddev=stddev2))

        conv = tf.nn.separable_conv2d(inputs, depthwise_filter=kernel1, pointwise_filter=kernel2, strides=strides,
                                      padding=padding, name="sep_conv")
        bias = tf.compat.v1.get_variable("biases", kernel_shape[3],
                                         initializer=tf.constant_initializer(value=biasInit))
        outputs = tf.nn.bias_add(conv, bias, name='preActivation')
        if activation:
            outputs = activation(outputs, name='activation')
        if drop_rate > 0.0:
            outputs = dropout(outputs, is_training=is_training, rate=drop_rate)
        return outputs


def dil_conv2d(inputs, is_training,
               kernel_shape,
               rate,
               activation=relu,
               drop_rate=0.0,
               initOpt=0, biasInit=0.1, padding="SAME", name='dil_conv2d'):
    """Adds a 2-D convolutional layer given 4-D `inputs` and `kernel` with optional BatchNorm, LocalResponseNorm and Dropout.

    Args:
        scope_or_name: `string` or `VariableScope`, the scope to open.
        inputs: `4-D Tensor`, it is assumed that `inputs` is shaped `[batch_size, Y, X, Z]`.
        kernel: `4-D Tensor`, [kernel_height, kernel_width, in_channels, out_channels] kernel.
        bias: `1-D Tensor`, [out_channels] bias.
        rate: `int`, Dilation factor.
        activation: activation function to be used (default: `relu`).
        use_bn: `bool`, whether or not to include batch normalization in the layer.
        is_training: `bool`, whether or not the layer is in training mode. This is only used if `use_bn` == True.
        use_lrn: `bool`, whether or not to include local response normalization in the layer.
        keep_prob: `double`, dropout keep prob.
        dropout_maps: `bool`, If true whole maps are dropped or not, otherwise single elements.
        padding: `string` from 'SAME', 'VALID'. The type of padding algorithm used in the convolution.

    Returns:
        `4-D Tensor`, has the same type `inputs`.
    """
    with tf.compat.v1.variable_scope(name):
        stddev = 5e-2
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])), 5e-2)
        initializer = tf.random_normal_initializer(stddev=stddev)
        if initOpt < 0:
            initializer = tf.random.truncated_normal_initializer(0.0, -initOpt)
        kernel = tf.compat.v1.get_variable("weights", kernel_shape,
                                           initializer=initializer)
        conv = tf.nn.atrous_conv2d(inputs, kernel, rate=rate, padding=padding)
        bias = tf.compat.v1.get_variable("bias", kernel_shape[3],
                                         initializer=tf.constant_initializer(value=biasInit))
        outputs = tf.nn.bias_add(conv, bias, name='preActivation')
        if activation:
            outputs = activation(outputs, name='activation')
        if drop_rate > 0.0:
            outputs = dropout(outputs, is_training=is_training, rate=drop_rate)
        return outputs


def deconv2d(inputs, is_training, kernel_shape, out_shape, subS=2, activation=relu,
             drop_rate=0.0,
             initOpt=0, biasInit=0.1, name='deconv2d'):
    with tf.compat.v1.variable_scope(name):
        stddev = 5e-2
        if initOpt == 0:
            stddev = np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2] + kernel_shape[3]))
        if initOpt == 1:
            stddev = 5e-2
        if initOpt == 2:
            stddev = min(np.sqrt(2.0 / (kernel_shape[0] * kernel_shape[1] * kernel_shape[2])), 5e-2)
        initializer = tf.random_normal_initializer(stddev=stddev)
        if initOpt < 0:
            initializer = tf.random.truncated_normal_initializer(0.0, -initOpt)
        kernel = tf.compat.v1.get_variable("weights", kernel_shape,
                                           initializer=initializer)
        bias = tf.compat.v1.get_variable("bias", kernel_shape[2],
                                         initializer=tf.constant_initializer(value=biasInit))
        conv = tf.nn.conv2d_transpose(inputs, kernel, out_shape, strides=[1, subS, subS, 1], padding='SAME',
                                      name='conv')
        outputs = tf.nn.bias_add(conv, bias, name='preActivation')
        if activation:
            outputs = activation(outputs, name='activation')
        if drop_rate > 0.0:
            outputs = dropout(outputs, is_training=is_training, rate=drop_rate)
        return outputs


# </editor-fold>

# <editor-fold desc="Recurrent Layer (b_rnn_layer)">
def b_rnn_layer(inputs,
                is_training,
                n_hidden,
                seq_length=None,
                use_cudnn=True,
                time_major=True,
                cell_type='LSTM',
                name='b_rnn'):
    """
    Bidirectional RNN layer. Input is assumed to be of form (dim0 x dim1 x FEATURES_IN) output is of form (dim0 x dim1 x FEATURES_OUT)
    :param inputs:
    :param is_training:
    :param n_hidden:
    :param seq_length:
    :param use_cudnn:
    :param time_major:
    :param cell_type: 'LSTM' or 'GRU'
    :param name:
    :return:
    """
    with tf.compat.v1.variable_scope(name):
        if use_cudnn:
            if not time_major:
                print('Time major false not supported for variable sequence length.')
            # forward direction
            with tf.compat.v1.variable_scope("culstm_forward"):
                if cell_type == 'LSTM':
                    curnn_fw = CudnnLSTM(num_layers=1, num_units=n_hidden, direction="unidirectional", dtype=tf.float32)
                if cell_type == 'GRU':
                    curnn_fw = CudnnGRU(num_layers=1, num_units=n_hidden, direction="unidirectional", dtype=tf.float32)
                curnn_fw.build(inputs.get_shape())
                outputs_fw, _ = curnn_fw(inputs, training=is_training)
                # culstm_fw = tf.keras.layers.CuDNNLSTM(units=n_hidden, return_sequences=True)
                # culstm_fw.build(inputs.get_shape())
                # outputs_fw = culstm_fw(inputs, training=is_training)
            # backward direction
            with tf.compat.v1.variable_scope("culstm_backward"):
                if cell_type == 'LSTM':
                    curnn_bw = CudnnLSTM(num_layers=1, num_units=n_hidden, direction="unidirectional", dtype=tf.float32)
                if cell_type == 'GRU':
                    curnn_bw = CudnnGRU(num_layers=1, num_units=n_hidden, direction="unidirectional", dtype=tf.float32)
                curnn_bw.build(inputs.get_shape())
                reverse_inputs = tf.reverse_sequence(inputs, seq_length, batch_axis=1, seq_axis=0)
                outputs_bw, _ = curnn_bw(reverse_inputs, training=is_training)
                outputs_bw = tf.reverse_sequence(outputs_bw, seq_length, batch_axis=1, seq_axis=0)
                # culstm_bw = tf.keras.layers.CuDNNLSTM(units=n_hidden, return_sequences=True)
                # culstm_bw.build(inputs.get_shape())
                # reverse_inputs = tf.reverse_sequence(inputs, seq_length, batch_axis=1, seq_axis=0)
                # outputs_bw = culstm_bw(reverse_inputs, training=is_training)
                # outputs_bw = tf.reverse_sequence(outputs_bw, seq_length, batch_axis=1, seq_axis=0)
            # concat
            outputs = tf.concat([outputs_fw, outputs_bw], axis=2)

        else:
            from tensorflow.python.ops.rnn import dynamic_rnn as rnn
            if cell_type == 'LSTM':
                single_cell = lambda: LSTMCell(n_hidden, reuse=tf.compat.v1.get_variable_scope().reuse)
            if cell_type == 'GRU':
                single_cell = lambda: GRUCell(n_hidden, reuse=tf.compat.v1.get_variable_scope().reuse)
            # forward direction
            with tf.compat.v1.variable_scope("culstm_forward"):
                cell_fw = MultiRNNCell([single_cell() for _ in range(1)])
                outputs_fw, _ = rnn(cell_fw, inputs, dtype=tf.float32, time_major=True)
            # backward direction
            with tf.compat.v1.variable_scope("culstm_backward"):
                cell_bw = MultiRNNCell([single_cell() for _ in range(1)])
                reverse_inputs = tf.reverse_sequence(inputs, seq_length, batch_axis=1, seq_axis=0)
                outputs_bw, _ = rnn(cell_bw, reverse_inputs, dtype=tf.float32, time_major=True)
                outputs_bw = tf.reverse_sequence(outputs_bw, seq_length, batch_axis=1, seq_axis=0)
            # concat
            outputs = tf.concat([outputs_fw, outputs_bw], axis=2)
    return outputs


def MultiRNNCell(cells, state_is_tuple=True):
    return tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=state_is_tuple)


def LSTMCell(num_units, reuse=None):
    return CudnnCompatibleLSTMCell(num_units=num_units, reuse=reuse)


def GRUCell(num_units, reuse=None):
    return CudnnCompatibleGRUCell(num_units=num_units, reuse=reuse)


def DropoutWrapper(cell, input_keep_prob=1.0, output_keep_prob=1.0, state_keep_prob=1.0):
    return tf.compat.v1.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=input_keep_prob,
                                                   output_keep_prob=output_keep_prob,
                                                   state_keep_prob=state_keep_prob)


# </editor-fold>

# <editor-fold desc="MLP (mlp)">
def mlp(inputs, num_hidden_units, output_dim, is_training,
        hidden_activation=relu, output_activation=None, use_bias=True, reuse=None,
        dropout_rate=0.0, dropout_output=False, name='mlp'):
    # inputs: [num_inputs, input_dim] float
    # num_hidden_units: list of scalar int of length num_hidden_layers
    # output_dim: scalar int

    with tf.compat.v1.variable_scope(name):
        # hidden layers
        for index, hidden_dim in enumerate(num_hidden_units):
            inputs = ff_layer(inputs, outD=hidden_dim, is_training=is_training,
                              activation=hidden_activation, use_bias=use_bias, reuse=reuse,
                              name='fully_connected_layer_h' + str(index + 1))
            if dropout_rate > 0.0:
                inputs = dropout(inputs, is_training, dropout_rate)
        # output layer
        outputs = ff_layer(inputs, outD=output_dim, is_training=is_training,
                           activation=output_activation, use_bias=use_bias, reuse=reuse,
                           name='fully_connected_logit_layer_out')
        if dropout_output and dropout_rate > 0.0:
            outputs = dropout(outputs, is_training, dropout_rate)
        # inputs: [num_inputs, output_dim]
        return outputs


# </editor-fold>

# </editor-fold>

# <editor-fold desc="NOT Trainable Layers (dropout, avg_pool1d, avg_pool2d, max_pool1d, max_pool2d, embedding_lookup, moments, top_k, conv_to_rnn, brnn_direction_merge_sum, brnn_direction_merge_sum_to_conv, normalize, upsample_simple, per_image_standardization)">
def dropout(inputs, is_training, rate=None, keep_prob=None, noise_shape=None, name="dropout"):
    if rate and keep_prob:
        print("ERROR: Use either keep_prob or rate for dropout! ")
        exit(1)
    if keep_prob:
        rate = 1.0 - keep_prob
    # if is_training:
    #     return tf.nn.dropout(inputs, rate=rate, noise_shape=noise_shape, name=name)
    # else:
    #     return inputs
    # Workaround for java based training.
    rate = tf.cast(is_training, dtype=tf.float32) * rate
    return tf.nn.dropout(inputs, rate=rate, noise_shape=noise_shape, name=name)


def avg_pool1d(inputs,
               kernel_width,
               stride_width,
               padding='SAME',
               name="avg_pool1d"):
    with tf.compat.v1.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=1)
        outputs = avg_pool2d(inputs, ksize=[1, 1, kernel_width, 1], strides=[1, 1, stride_width, 1],
                             padding=padding)
        outputs = tf.squeeze(outputs, axis=1)
        return outputs


def avg_pool2d(inputs, ksize, strides, padding, name='avg_pool2d'):
    return tf.nn.avg_pool2d(inputs, ksize=ksize, strides=strides, padding=padding, name=name)


def max_pool1d(inputs,
               kernel_width,
               stride_width,
               padding='SAME',
               name="max_pool1d"):
    with tf.compat.v1.variable_scope(name):
        inputs = tf.expand_dims(inputs, axis=1)
        outputs = max_pool2d(inputs, ksize=[1, 1, kernel_width, 1], strides=[1, 1, stride_width, 1],
                             padding=padding)
        outputs = tf.squeeze(outputs, axis=1)
        return outputs


def max_pool2d(inputs, ksize, strides, padding, name='max_pool2d'):
    return tf.nn.max_pool2d(inputs, ksize=ksize, strides=strides, padding=padding, name=name)


def embedding_lookup(params,
                     ids,
                     partition_strategy="mod",
                     name=None,
                     validate_indices=True,  # pylint: disable=unused-argument
                     max_norm=None):
    return tf.nn.embedding_lookup(params, ids, partition_strategy=partition_strategy, name=name,
                                  validate_indices=validate_indices, max_norm=max_norm)


def moments(x, axes, shift=None, name=None, keep_dims=None, keepdims=None):
    return tf.nn.moments(x,
                         axes,
                         shift=shift,  # pylint: disable=unused-argument
                         name=name,
                         keep_dims=keep_dims,
                         keepdims=keepdims)


def top_k(input, k=1, sorted=True, name=None):
    return tf.nn.top_k(input, k=k, sorted=sorted, name=name)


def conv_to_rnn(conv_out, time_major=True, data_format='NHWC', name='conv_to_rnn'):
    """
    Adds a utility layer to transform the output Tensor of a convolutional layer into a Tensor which fits a RNN.
    This function assumes that `time_major` fits time major of RNN.

    Args:
        conv_out: `4-D Tensor`, the output of a convolutional layer with its shaped determined by `data_format`:

                  For 'NHWC' it is assumed that `conv_out` is stored in the order of `[batch_size, Y, X, Z]`.

                  For 'NCHW' it is assumed that `conv_out` is stored in the order of `[batch_size, Z, Y, X]`.
        time_major: `bool` [batch_size, time, depth] vs [time, batch_size, depth].
        data_format: `str` from ('NHWC', 'NCHW'), specifies the data format of `conv_out`.
        name: `str` or `VariableScope`, the scope to open.

    Returns:
        `3-D Tensor`, the transformed `conv_out` Tensor with shape `[X, batch_size, Y * Z]` corresponds to
        `[max_time, batch_size, cell_size]` for time_major=True.
    """
    with tf.compat.v1.variable_scope(name):
        if data_format == 'NCHW':
            if time_major:
                # (batch_size, Z, Y, X) -> (X, batch_size, Y, Z)
                rnn_in = tf.transpose(conv_out, [3, 0, 2, 1])
            else:
                # (batch_size, Z, Y, X) -> (batch_size, X, Y, Z)
                rnn_in = tf.transpose(conv_out, [0, 3, 2, 1])
        else:
            if time_major:
                # (batch_size, Y, X, Z) -> (X, batch_size, Y, Z)
                rnn_in = tf.transpose(conv_out, [2, 0, 1, 3])
            else:
                # (batch_size, Y, X, Z) -> (batch_size, X, Y, Z)
                rnn_in = tf.transpose(conv_out, [0, 2, 1, 3])

        shape_static = rnn_in.get_shape().as_list()
        y = shape_static[2]
        z = shape_static[3]
        shape_dynamic = tf.shape(rnn_in)
        dim0 = shape_dynamic[0]
        dim1 = shape_dynamic[1]
        # (dim0, dim1, Y, Z) -> (dim0, dim1, Y*Z)
        rnn_in = tf.reshape(rnn_in, [dim0, dim1, y * z])
        # (X, batch_size, Y*Z) corresponds to [max_time, batch_size, cell_size]
    return rnn_in


def brnn_direction_merge_sum(rnn_out, time_major_in=True, time_major_out=True, name='brnn_merge_sum'):
    """
    Adds a utility layer to transform the output Tensor pair of a bidirectional dynamic RNN into a 3D Tensor,
    which sums the both RNN directions
    """
    with tf.compat.v1.variable_scope(name):
        shape_static = rnn_out.get_shape().as_list()
        cell_size = shape_static[2] // 2
        shape_dynamic = tf.shape(rnn_out)
        dim0 = shape_dynamic[0]
        dim1 = shape_dynamic[1]
        # [dim0, dim1, 2*cell_size] -> [dim0, dim1, 2, cell_size]
        graph_o = tf.reshape(rnn_out, shape=[dim0, dim1, 2, cell_size])
        # [dim0, dim1, 2, cell_size] -> [dim0, dim1, cell_size]
        graph_o = tf.reduce_sum(graph_o, axis=2)
        if time_major_in and time_major_out:
            return graph_o
        else:
            # Since time_major_in != time_major_out we flip the first two dimensions
            return tf.transpose(graph_o, [1, 0, 2])


def brnn_direction_merge_sum_to_conv(rnn_out, time_major_in=True, data_format='NHWC', name='brnn_merge_sum_to_conv'):
    with tf.compat.v1.variable_scope(name):
        # [] -> [batch_size, max_time, cell_size]
        output = brnn_direction_merge_sum(rnn_out, time_major_in, time_major_out=False)
        # [batch_size, max_time, cell_size] -> [batch_size, cell_size, max_time]
        output = tf.transpose(output, [0, 2, 1])
        if data_format == 'NHWC':
            # [batch_size, cell_size, max_time] -> [batch_size, cell_size, max_time, 1]
            return tf.expand_dims(output, axis=3)
            # [batch_size, cell_size, max_time, 1] corresponds to [batch_size, Y, X, Z], 'NHWC'
        else:
            # [batch_size, cell_size, max_time] -> [batch_size, 1, cell_size, max_time]
            return tf.expand_dims(output, axis=1)
            # [batch_size, 1, cell_size, max_time] corresponds to [batch_size, Z, Y, X], 'NCHW'


def normalize(image, img_length):
    # dynamic shape values (calculated during runtime)
    shape_dynamic = tf.shape(image)
    # static shape values (defined up-front)
    shape_static = image.get_shape().as_list()
    # image normalization
    image_crop = tf.image.crop_to_bounding_box(image, 0, 0, shape_static[0], img_length)
    # image_norm = tf.image.per_image_standardization(image_crop)
    image_norm = per_image_standardization(image_crop)
    # TODO test this
    # image_norm = 1.0 - image_norm
    image_pad = tf.image.pad_to_bounding_box(image_norm, 0, 0, shape_static[0], shape_dynamic[1])
    # image_pad = 1.0 - image_pad
    return image_pad


@tf_export('image.per_image_standardization')
def per_image_standardization(image):
    """Linearly scales `image` to have zero mean and unit norm.

    This op computes `(x - mean) / adjusted_stddev`, where `mean` is the average
    of all values in image, and
    `adjusted_stddev = max(stddev, 1.0/sqrt(image.NumElements()))`.

    `stddev` is the standard deviation of all values in `image`. It is capped
    away from zero to protect against division by 0 when handling uniform images.

    Args:
      image: 3-D tensor of shape `[height, width, channels]`.

    Returns:
      The standardized image with same shape as `image`.

    Raises:
      ValueError: if the shape of 'image' is incompatible with this function.
    """
    with tf.name_scope(None, 'per_image_standardization', [image]) as scope:
        image = tf.convert_to_tensor(image, name='image')
        # image = _Assert3DImage(image)
        # num_pixels = math_ops.reduce_prod(array_ops.shape(image))

        # image = math_ops.cast(image, dtype=dtypes.float32)
        image_mean = tf.math.reduce_mean(image)

        variance = (
                tf.math.reduce_mean(tf.math.square(image)) -
                tf.math.square(image_mean))
        variance = relu(variance)
        stddev = tf.math.sqrt(variance)

        # Apply a minimum normalization that protects us against uniform images.
        # min_stddev = math_ops.rsqrt(1.0 * num_pixels)
        pixel_value_scale = tf.math.maximum(stddev, 0.0001)
        pixel_value_offset = image_mean

        image = tf.math.subtract(image, pixel_value_offset)
        image = tf.math.divide(image, pixel_value_scale, name=scope)
        # image = math_ops.div(image, pixel_value_scale, name=scope)
        return image


def upsample_simple(images, shape_out, up, numClasses):
    filter_up = tf.constant(1.0, shape=[up, up, numClasses, numClasses])
    return tf.nn.conv2d_transpose(images, filter_up,
                                  output_shape=shape_out,
                                  strides=[1, up, up, 1])


# </editor-fold>

# <editor-fold desc="Loss Utilities (softmax_cross_entropy_with_logits_v2, sparse_softmax_cross_entropy_with_logits, sigmoid_cross_entropy_with_logits, l2_loss, nce_loss)">
def softmax_cross_entropy_with_logits_v2(labels, logits, axis=None, name=None, dim=None):
    return tf.nn.softmax_cross_entropy_with_logits_v2(labels, logits, axis=axis, name=name, dim=dim)


def sparse_softmax_cross_entropy_with_logits(labels=None, logits=None, name=None, _sentinel=None):
    return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits, name=name, _sentinel=_sentinel)


def sigmoid_cross_entropy_with_logits(labels=None, logits=None, name=None, _sentinel=None):
    return tf.nn.sigmoid_cross_entropy_with_logits(_sentinel=_sentinel, labels=labels, logits=logits, name=name)


def l2_loss(t, name=None):
    return tf.nn.l2_loss(t, name=name)


def nce_loss(weights, biases, labels, inputs, num_sampled, num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss"):
    return tf.nn.nce_loss(weights,
                          biases,
                          labels,
                          inputs,
                          num_sampled,
                          num_classes,
                          num_true=num_true,
                          sampled_values=sampled_values,
                          remove_accidental_hits=remove_accidental_hits,
                          partition_strategy=partition_strategy,
                          name=name)


# </editor-fold>

# <editor-fold desc="Utilities (conv1d_op, conv2d_op)">
def conv1d_op(value=None,
              filters=None,
              stride=None,
              padding=None,
              use_cudnn_on_gpu=None,
              data_format=None,
              name=None,
              input=None,  # pylint: disable=redefined-builtin
              dilations=None):
    return tf.nn.conv1d(value=value,
                        filters=filters,
                        stride=stride,
                        padding=padding,
                        use_cudnn_on_gpu=use_cudnn_on_gpu,
                        data_format=data_format,
                        name=name,
                        input=input,
                        dilations=dilations)


def conv2d_op(input,
              filter=None,
              strides=None,
              padding=None,
              use_cudnn_on_gpu=True,
              data_format="NHWC",
              dilations=[1, 1, 1, 1],
              name=None,
              filters=None):
    return tf.nn.conv2d(input,
                        filter=filter,
                        strides=strides,
                        padding=padding,
                        use_cudnn_on_gpu=use_cudnn_on_gpu,
                        data_format=data_format,
                        dilations=dilations,
                        name=name,
                        filters=filters)

# </editor-fold>
