import tensorflow as tf
import numpy as np
from .encoder import Encoder
from ..utils.blocks import conv_lstm
from ..utils.attention import residual, layer_norm


class CONV_LSTM(Encoder):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''

    def encode(self, features, len_features):

        num_hidden = self.args.model.encoder.num_hidden
        use_residual = self.args.model.encoder.use_residual
        dropout = self.args.model.encoder.dropout
        num_filters = self.args.model.encoder.num_filters
        size_feat = self.args.data.dim_input

        # x = tf.expand_dims(features, -1)
        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        # size_feat = int(size_feat/3)
        len_feats = tf.reduce_sum(tf.cast(tf.reduce_sum(tf.abs(features), -1) > 0, tf.int32), -1)
        # x = tf.reshape(features, [size_batch, size_length, size_feat, 3])
        x = tf.reshape(features, [size_batch, size_length, size_feat, 1])
        # the first cnn layer
        x = self.normal_conv(
            inputs=x,
            filter_num=num_filters,
            kernel=(3,3),
            stride=(2,2),
            padding='SAME',
            use_relu=True,
            name="conv",
            w_initializer=None,
            norm_type='layer')
        x = conv_lstm(
            x=x,
            kernel_size=(3,3),
            filters=num_filters)

        size_feat = int(np.ceil(size_feat/2))*num_filters
        size_length  = tf.cast(tf.math.ceil(tf.cast(size_length,tf.float32)/2), tf.int32)
        len_seq = tf.cast(tf.math.ceil(tf.cast(len_feats, tf.float32)/2), tf.int32)
        x = tf.reshape(x, [size_batch, size_length, size_feat])

        outputs = x

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            num_hidden=num_hidden,
            use_residual=use_residual,
            dropout=dropout,
            name='blstm_1')
        outputs, len_seq = self.pooling(outputs, len_seq, 'HALF', 1)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            num_hidden=num_hidden,
            use_residual=use_residual,
            dropout=dropout,
            name='blstm_2')
        outputs, len_seq = self.pooling(outputs, len_seq, 'SAME', 2)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            num_hidden=num_hidden,
            use_residual=use_residual,
            dropout=dropout,
            name='blstm_3')
        outputs, len_seq = self.pooling(outputs, len_seq, 'HALF', 3)

        outputs = self.blstm(
            hidden_output=outputs,
            len_feas=len_seq,
            num_hidden=num_hidden,
            use_residual=use_residual,
            dropout=dropout,
            name='blstm_4')
        outputs, len_seq = self.pooling(outputs, len_seq, 'SAME', 4)

        pad_mask = tf.tile(tf.expand_dims(tf.sequence_mask(len_seq, tf.shape(outputs)[1], tf.float32), -1),
                           [1, 1, num_hidden])
        outputs *= pad_mask

        return outputs, len_seq

    @staticmethod
    def normal_conv(inputs, filter_num, kernel, stride, padding, use_relu, name,
                    w_initializer=None, norm_type="batch"):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding,
                               kernel_initializer=w_initializer, name="conv")
            if norm_type == "batch":
                net = tf.layers.batch_normalization(net, name="bn")
            elif norm_type == "layer":
                net = layer_norm(net)
            else:
                net = net
            output = tf.nn.relu(net) if use_relu else net

        return output

    @staticmethod
    def blstm(hidden_output, len_feas, num_hidden, use_residual, dropout, name):
        num_hidden /= 2

        with tf.variable_scope(name):
            f_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden)
            b_cell = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(num_hidden)

            x, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=f_cell,
                cell_bw=b_cell,
                inputs=hidden_output,
                dtype=tf.float32,
                time_major=False,
                sequence_length=len_feas)
            x = tf.concat(x, 2)

            if use_residual:
                x = residual(hidden_output, x, dropout)

        return x

    def pooling(self, x, len_sequence, type, name):
        num_hidden = self.args.model.encoder.num_hidden

        x = tf.expand_dims(x, axis=2)
        x = self.normal_conv(
            x,
            num_hidden,
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
