import tensorflow as tf
import numpy as np
from .encoder import Encoder
from ..utils.blocks import conv_lstm
from ..utils.attention import layer_norm


class CONV_LSTM(Encoder):

    def __init__(self, args, training, name='conv_lstm'):
        self.hidden_size = args.model.encoder.hidden_size
        self.num_filters = args.model.encoder.num_filters
        self.feature_map = self.args.data.num_feat_map
        super().__init__(args, training, name)

    def __call__(self, features, len_features):

        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        size_feat = int(self.args.data.dim_input / self.feature_map)
        x = tf.reshape(features, [size_batch, size_length, size_feat, self.feature_map])

        # the first cnn layer
        x = self.normal_conv(
            inputs=x,
            filter_num=self.num_filters,
            kernel=(3,3),
            stride=(2,2),
            padding='SAME',
            use_relu=True,
            name="conv",
            norm_type='layer')
        x = conv_lstm(
            x=x,
            kernel_size=(3,3),
            filters=self.num_filters)

        size_feat = int(np.ceil(size_feat/2)) * self.num_filters
        size_length  = tf.cast(tf.math.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
        len_seq = tf.cast(tf.math.ceil(tf.cast(len_features, tf.float32)/2), tf.int32)
        x = tf.reshape(x, [size_batch, size_length, size_feat])

        outputs = x

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            hidden_size=self.hidden_size,
            name='blstm_1')
        outputs, len_seq = self.pooling(outputs, len_seq, 'HALF', 1)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            hidden_size=self.hidden_size,
            name='blstm_2')
        outputs, len_seq = self.pooling(outputs, len_seq, 'SAME', 2)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            hidden_size=self.hidden_size,
            name='blstm_3')
        outputs, len_seq = self.pooling(outputs, len_seq, 'HALF', 3)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            hidden_size=self.hidden_size,
            name='blstm_4')
        outputs, len_seq = self.pooling(outputs, len_seq, 'SAME', 4)

        pad_mask = tf.tile(tf.sequence_mask(len_seq, tf.shape(outputs)[1], tf.float32)[:, :, None],
                           [1, 1, self.hidden_size])
        outputs *= pad_mask

        return outputs, len_seq

    @staticmethod
    def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name, norm_type="batch"):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding, name="conv")
            if norm_type == "batch":
                net = tf.layers.batch_normalization(net, name="bn")
            elif norm_type == "layer":
                net = layer_norm(net)
            else:
                net = net
            output = tf.nn.relu(net) if use_relu else net

        return output

    @staticmethod
    def blstm(hidden_output, len_feas, hidden_size, name):
        hidden_size /= 2

        with tf.variable_scope(name):
            f_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)
            b_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(hidden_size)

            x, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=f_cell,
                cell_bw=b_cell,
                inputs=hidden_output,
                dtype=tf.float32,
                time_major=False,
                sequence_length=len_feas)
            x = tf.concat(x, 2)

        return x

    def pooling(self, x, len_sequence, type, name):
        hidden_size = self.args.model.encoder.hidden_size

        x = tf.expand_dims(x, axis=2)
        x = self.normal_conv(
            x,
            hidden_size,
            (1, 1),
            (1, 1),
            'SAME',
            'True',
            name="tdnn_"+str(name),
            norm_type='layer')

        if type == 'SAME':
            x = tf.layers.max_pooling2d(x, (1, 1), (1, 1), 'SAME')
        elif type == 'HALF':
            x = tf.layers.max_pooling2d(x, (2, 1), (2, 1), 'SAME')
            len_sequence = tf.cast(tf.math.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)

        x = tf.squeeze(x, axis=2)

        return x, len_sequence
