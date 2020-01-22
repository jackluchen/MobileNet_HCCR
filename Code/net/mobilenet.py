import tensorflow as tf
from nets.ops import conv, expanded_conv, DepthSepConv, global_pool, make_divisible


class MobileNetV1:
    def __init__(self, n_classes, depth_rate=1.0, is_training=True):
        self.n_classes = n_classes
        self.depth_rate = depth_rate
        self.is_training = is_training

    def depth(self, n, rate, min_depth=8):
        return max(int(n * rate), min_depth)

    def parseNet(self, x, layer_list):
        # layer:[op,kernel_size,stride,out_channels,depth]
        for i, layer in enumerate(layer_list):
            op, k, s, out_c = layer
            out_c = self.depth(out_c, self.depth_rate)
            # op(input,name,k,s,out_c,is_training)
            x = op(x, 'Conv_%d' % i, k=k, s=s, out_c=out_c, is_training=self.is_training)
        return x

    def build_graph(self, input_tf):
        layer_list = [[conv, 3, 2, 32],
                      [DepthSepConv, 3, 1, 64],
                      [DepthSepConv, 3, 2, 128],
                      [DepthSepConv, 3, 1, 128],
                      [DepthSepConv, 3, 2, 256],
                      [DepthSepConv, 3, 1, 256],
                      [DepthSepConv, 3, 2, 512],
                      [DepthSepConv, 3, 1, 512],
                      [DepthSepConv, 3, 1, 512],
                      [DepthSepConv, 3, 1, 512],
                      [DepthSepConv, 3, 1, 512],
                      [DepthSepConv, 3, 1, 512],
                      [DepthSepConv, 3, 2, 1024],
                      [DepthSepConv, 3, 1, 1024],
                      ]
        x = self.parseNet(input_tf, layer_list)
        x = global_pool(x)
        if self.is_training:
            x = tf.nn.dropout(x, keep_prob=0.50)
        x = conv(x, 'Conv_20', k=1, s=1, out_c=self.n_classes, with_bias=True, use_bn=False, relu=None,
                 is_training=self.is_training)
        x = tf.squeeze(x, [1, 2])
        return x


class MobileNetV2:
    def __init__(self, n_classes, depth_rate=1.0, is_training=True):
        self.n_classes = n_classes
        self.depth_rate = depth_rate
        self.is_training = is_training

    def depth(self, n, rate, min_depth=8):
        return make_divisible(n * rate, divisor=8, min_value=min_depth)

    def parseNet(self, x, layer_list):
        # layer:[op,kernel_size,stride,out_channels]
        for i, layer in enumerate(layer_list):
            if len(layer) == 4:
                op, k, s, out_c = layer
            elif len(layer) == 5:
                op, k, s, out_c, expand_rate = layer
            else:
                raise "Unknow layer=" + str(layer)
            out_c = self.depth(out_c, self.depth_rate)
            if op == conv:
                x = conv(x, 'Conv_%d' % i, k=k, s=s, out_c=out_c, is_training=self.is_training)
            elif op == expanded_conv:
                x = expanded_conv(x, 'Conv_%d' % i, k=k, s=s, out_c=out_c, expand_rate=expand_rate,
                                  is_training=self.is_training)
            else:
                raise "Unkenow Op=" + str(op)
        return x

    def build_graph(self, input_tf):
        # x = conv(input_tf,'Conv_1',k=3,s=2,out_c= self.depth(32,depth_rate))
        # x = expanded_conv(x,'Conv_2',k=3,s=1,out_c=self.depth(16,depth_rate),expand_rate=1,is_training=is_training)
        # x = expanded_conv(x,'Conv_3',k=3,s=2,out_c=self.depth(24,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_4',k=3,s=1,out_c=self.depth(24,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_5',k=3,s=2,out_c=self.depth(32,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_6',k=3,s=1,out_c=self.depth(32,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_7',k=3,s=1,out_c=self.depth(32,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_8',k=3,s=2,out_c=self.depth(64,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_9',k=3,s=1,out_c=self.depth(64,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_10',k=3,s=1,out_c=self.depth(64,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_11',k=3,s=1,out_c=self.depth(64,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_12',k=3,s=1,out_c=self.depth(96,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_13',k=3,s=1,out_c=self.depth(96,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_14',k=3,s=1,out_c=self.depth(96,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_15',k=3,s=2,out_c=self.depth(160,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_16',k=3,s=1,out_c=self.depth(160,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_17',k=3,s=1,out_c=self.depth(160,depth_rate),expand_rate=6,is_training=is_training)
        # x = expanded_conv(x,'Conv_18',k=3,s=1,out_c=self.depth(320,depth_rate),expand_rate=6,is_training=is_training)
        # x = conv(x,'Conv_19',k=1,s=1,out_c=self.depth(1280,depth_rate),is_training=is_training)
        layer_list = [[conv, 3, 2, 32],
                      [expanded_conv, 3, 1, 16, 1],
                      [expanded_conv, 3, 2, 24, 6],
                      [expanded_conv, 3, 1, 24, 6],
                      [expanded_conv, 3, 2, 32, 6],
                      [expanded_conv, 3, 1, 32, 6],
                      [expanded_conv, 3, 1, 32, 6],
                      [expanded_conv, 3, 2, 64, 6],
                      [expanded_conv, 3, 1, 64, 6],
                      [expanded_conv, 3, 1, 64, 6],
                      [expanded_conv, 3, 1, 64, 6],
                      [expanded_conv, 3, 1, 96, 6],
                      [expanded_conv, 3, 1, 96, 6],
                      [expanded_conv, 3, 1, 96, 6],
                      [expanded_conv, 3, 2, 160, 6],
                      [expanded_conv, 3, 1, 160, 6],
                      [expanded_conv, 3, 1, 160, 6],
                      [expanded_conv, 3, 1, 320, 6],
                      [conv, 1, 1, 1280]]
        x = self.parseNet(input_tf, layer_list)
        x = global_pool(x)
        if self.is_training:
            x = tf.nn.dropout(x, keep_prob=0.50)
        x = conv(x, 'Conv_20', k=1, s=1, out_c=self.n_classes, with_bias=True, use_bn=False, relu=None,
                 is_training=self.is_training)
        x = tf.squeeze(x, [1, 2])
        return x
