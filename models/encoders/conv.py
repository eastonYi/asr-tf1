'''@file listener.py
contains the listener code'''

import tensorflow as tf

from .encoder import Encoder
from ..utils.blocks import normal_conv, block
# class CONV(Encoder):
#     '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
#     '''
#
#     def encode(self, features, len_feas):
#         num_filters = self.args.model.encoder.num_filters
#         size_feat = self.args.data.dim_input
#
#         # x = tf.expand_dims(features, -1)
#         size_batch  = tf.shape(features)[0]
#         size_length = tf.shape(features)[1]
#         x = tf.reshape(features, [size_batch, size_length, size_feat, 1])
#
#         for i in range(3):
#             x = self.normal_conv(
#                 inputs=x,
#                 filter_num=num_filters,
#                 kernel=(3,3),
#                 stride=(2,1),
#                 padding='SAME',
#                 use_relu=True,
#                 name="conv_"+str(i),
#                 norm_type='layer')
#             size_length = tf.cast(tf.ceil(tf.cast(size_length,tf.float32)/2), tf.int32)
#
#         x = conv_lstm(
#             inputs=x,
#             kernel_size=(3,3),
#             filters=num_filters)
#
#         output_seq_lengths = tf.cast(tf.ceil(tf.cast(len_feas,tf.float32)/2), tf.int32)
#         outputs = tf.reshape(x, [size_batch, size_length, num_filters*size_feat])
#
#         return outputs, output_seq_lengths
#
#     @staticmethod
#     def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name,
#                     w_initializer=None, norm_type="batch"):
#         with tf.variable_scope(name):
#             net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding,
#                                kernel_initializer=w_initializer, name="conv")
#             if norm_type == "batch":
#                 net = tf.layers.batch_normalization(net, name="bn")
#             elif norm_type == "layer":
#                 net = layer_norm(net)
#             else:
#                 net = net
#             output = tf.nn.relu(net) if use_relu else net
#
#         return output
class CONV(Encoder):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''

    def encode(self, features, len_feas):
        num_filters = self.args.model.encoder.num_filters
        size_feat = self.args.data.dim_input

        # x = tf.expand_dims(features, -1)
        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        x = tf.reshape(features, [size_batch, size_length, size_feat, 1])

        for i in range(2):
            x = normal_conv(
                inputs=x,
                filter_num=num_filters,
                kernel=(3,3),
                stride=(2,1),
                padding='SAME',
                use_relu=True,
                name="conv_"+str(i),
                norm_type='layer')
            size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
            len_feas = tf.cast(tf.ceil(tf.cast(len_feas, tf.float32)/2), tf.int32)

        for i in range(10):
            x = block(x, num_filters, i)

        outputs = tf.reshape(x, [size_batch, size_length, num_filters*size_feat])
        outputs *= tf.sequence_mask(len_feas,
                                    maxlen=tf.shape(outputs)[1],
                                    dtype=tf.float32)[:, : , None]

        return outputs, len_feas
