import tensorflow as tf


def conv(in_tensor, layer_name, k, s, out_c, is_training=True, with_bias=False, use_bn=True, relu=tf.nn.relu6):
    with tf.variable_scope(layer_name):
        in_size = in_tensor.get_shape().as_list()
        ss = [1, s, s, 1]
        kernel_shape = [k, k, in_size[3], out_c]
        # conv
        kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                 tf.contrib.layers.xavier_initializer_conv2d(), trainable=is_training,
                                 collections=['wd', 'variables', 'filters'])

        x = tf.nn.conv2d(in_tensor, kernel, ss, padding='SAME')
        print(x)
        # bias
        if with_bias:
            biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                     tf.constant_initializer(0.0001), trainable=is_training,
                                     collections=['wd', 'variables', 'biases'])
            x = tf.nn.bias_add(x, biases, name='bias')
        if use_bn:
            x = batch_norm(x, is_training=is_training)
        if not relu is None:
            x = relu(x, "Relu6")
        return x


def batch_norm(in_tensor, is_training):
    return tf.contrib.layers.batch_norm(in_tensor,
                                        center=True,
                                        decay=0.9997,
                                        scale=True,
                                        epsilon=0.001,
                                        activation_fn=None,
                                        # updates_collections=None,
                                        is_training=is_training)


def max_pool(in_tensor, name='max_pool'):
    pooled = tf.nn.max_pool(in_tensor, ksize=[1, 2, 2, 1], ss=[1, 2, 2, 1],
                            padding='VALID', name=name)
    print(pooled)
    return pooled


def global_pool(input_tensor, pool_op=tf.nn.avg_pool):
    """Applies avg pool to produce 1x1 output.
    NOTE: This function is funcitonally equivalenet to reduce_mean, but it has
    baked in average pool which has better support across hardware.
    Args:
      input_tensor: input tensor
      pool_op: pooling op (avg pool is default)
    Returns:
      a tensor batch_size x 1 x 1 x depth.
    """
    shape = input_tensor.get_shape().as_list()
    if shape[1] is None or shape[2] is None:
        k = tf.convert_to_tensor(
            [1, tf.shape(input_tensor)[1],
             tf.shape(input_tensor)[2], 1])
    else:
        k = [1, shape[1], shape[2], 1]
    output = pool_op(
        input_tensor, ksize=k, strides=[1, 1, 1, 1], padding='VALID')
    # Recover output shape, for unknown shape.
    output.set_shape([None, 1, 1, None])
    return output


# Depthwise+Pointwise
def DepthSepConv(in_tensor, layer_name, k, s, out_c, is_training=True, relu=tf.nn.relu6):
    with tf.variable_scope(layer_name + '_depthwise'):
        in_size = in_tensor.get_shape().as_list()
        # depwise conv
        ss = [1, s, s, 1]
        # dw_kernel_shape=[filter_height, filter_width, in_channels, channel_multiplier]
        dw_kernel_shape = [k, k, in_size[3], 1]
        dw_kernel = tf.get_variable('depthwise_weights', dw_kernel_shape, tf.float32,
                                    tf.contrib.layers.xavier_initializer_conv2d(), trainable=is_training,
                                    collections=['wd', 'variables', 'filters'])

        x = tf.nn.depthwise_conv2d(in_tensor, dw_kernel, ss, padding='SAME', name=layer_name + '_depthwise')
        x = batch_norm(x, is_training=is_training)
        x = relu(x, "Relu6")
        print(x)
    with tf.variable_scope(layer_name + '_pointwise'):
        # pointwise conv
        pw_kernel_shape = [1, 1, in_size[3], out_c]
        pw_kernel = tf.get_variable('weights', pw_kernel_shape, tf.float32,
                                    tf.contrib.layers.xavier_initializer_conv2d(), trainable=is_training,
                                    collections=['wd', 'variables', 'filters'])

        x = tf.nn.conv2d(x, pw_kernel, [1, 1, 1, 1], padding='SAME')
        x = batch_norm(x, is_training=is_training)
        x = relu(x, "Relu6")
        print(x)
        return x


"""
MobileNet_V2中的Block实现，
代码参考自https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v2.py
"""


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


# mobile net V2
def expanded_conv(in_tensor, layer_name, k, s, out_c, expand_rate, is_training=True, relu=tf.nn.relu6):
    in_size = in_tensor.get_shape().as_list()
    expansion_size = make_divisible(in_size[3] * expand_rate, 8)
    x = in_tensor
    if expansion_size > in_size[3]:
        with tf.variable_scope(layer_name + '_expand'):
            # pointwise conv
            pw_kernel_shape = [1, 1, in_size[3], expansion_size]
            pw_kernel = tf.get_variable('weights', pw_kernel_shape, tf.float32,
                                        tf.contrib.layers.xavier_initializer_conv2d(), trainable=is_training,
                                        collections=['wd', 'variables', 'filters'])

            x = tf.nn.conv2d(in_tensor, pw_kernel, [1, 1, 1, 1], padding='SAME')
            x = batch_norm(x, is_training=is_training)
            x = relu(x, "Relu6")
            print(x)
    with tf.variable_scope(layer_name + '_depthwise'):

        # depwise conv
        ss = [1, s, s, 1]
        # dw_kernel_shape=[filter_height, filter_width, in_channels, channel_multiplier]
        dw_kernel_shape = [k, k, x.get_shape().as_list()[3], 1]
        dw_kernel = tf.get_variable('depthwise_weights', dw_kernel_shape, tf.float32,
                                    tf.contrib.layers.xavier_initializer_conv2d(), trainable=is_training,
                                    collections=['wd', 'variables', 'filters'])

        x = tf.nn.depthwise_conv2d(x, dw_kernel, ss, padding='SAME', name=layer_name + '_depthwise')
        x = batch_norm(x, is_training=is_training)
        x = relu(x, "Relu6")
        print(x)

    with tf.variable_scope(layer_name + '_project'):
        # pointwise conv
        pw_kernel_shape = [1, 1, x.get_shape().as_list()[3], out_c]
        pw_kernel = tf.get_variable('weights', pw_kernel_shape, tf.float32,
                                    tf.contrib.layers.xavier_initializer_conv2d(), trainable=is_training,
                                    collections=['wd', 'variables', 'filters'])

        x = tf.nn.conv2d(x, pw_kernel, [1, 1, 1, 1], padding='SAME')
        x = batch_norm(x, is_training=is_training)
        print(x)
    if s == 1 and out_c == in_size[3]:
        return in_tensor + x
    return x
