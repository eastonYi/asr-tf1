'''@file listener.py
contains the listener code'''

import tensorflow as tf

from .encoder import Encoder
from ..utils.blocks import normal_conv, block


class CONV(Encoder):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''

    def encode(self, features, len_feas):
        num_blocks = self.args.model.encoder.num_blocks
        num_filters = self.args.model.encoder.num_filters
        # num_blocks = self.args.model.encoder.num_blocks
        size_feat = int(self.args.data.dim_input/3)
        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        # x = tf.reshape(features, [size_batch, size_length, size_feat, 3])

        # for i in range(num_blocks):
        #     x = normal_conv(
        #         inputs=x,
        #         filter_num=num_filters,
        #         kernel=(3,9),
        #         stride=(2,1),
        #         padding='SAME',
        #         use_relu=True,
        #         name="conv_"+str(i),
        #         norm_type=None
        #         )
        #     size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
        #     len_feas = tf.cast(tf.ceil(tf.cast(len_feas, tf.float32)/2), tf.int32)

        # outputs = tf.reshape(x, [size_batch, size_length, num_filters*size_feat])

        x = tf.layers.dense(features, units=num_filters, use_bias=True)
        for i in range(num_blocks):
            inputs = x
            x = tf.layers.conv1d(x, filters=num_filters, kernel_size=5, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(x, filters=num_filters, kernel_size=5, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = inputs + 0.3*x
            x = tf.layers.max_pooling1d(x, 2, 2, 'same')
            size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
            len_feas = tf.cast(tf.ceil(tf.cast(len_feas, tf.float32)/2), tf.int32)

        outputs = tf.reshape(x, [size_batch, size_length, num_filters])

        # for i in range(num_fc):
        #     x = tf.layers.dense(x, units=hidden_size, use_bias=True)
        #     x = tf.nn.relu(x)

        outputs *= tf.sequence_mask(len_feas,
                              maxlen=tf.shape(outputs)[1],
                              dtype=tf.float32)[:, : , None]

        return outputs, len_feas
