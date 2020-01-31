'''@file listener.py
contains the listener code'''

import tensorflow as tf

from .encoder import Encoder
from ..utils.blocks import normal_conv, block, blstm
from ..utils.tfAudioTools import batch_splice

class CONV_1D(Encoder):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''
    def __init__(self, args, training, embed_table=None, name='conv1d'):
        self.num_blocks = args.model.encoder.num_blocks
        self.num_filters = args.model.encoder.num_filters
        super().__init__(args, training, embed_table, name)

    def encode(self, features, len_feas):

        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]

        x = tf.layers.dense(features, units=self.num_filters, use_bias=True)
        for i in range(self.num_blocks):
            inputs = x
            x = tf.layers.conv1d(x, filters=self.num_filters, kernel_size=5, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(x, filters=self.num_filters, kernel_size=5, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = inputs + 0.3*x
            x = tf.layers.max_pooling1d(x, 2, 2, 'same')
            size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
            len_feas = tf.cast(tf.ceil(tf.cast(len_feas, tf.float32)/2), tf.int32)

        outputs = tf.reshape(x, [size_batch, size_length, self.num_filters])

        outputs *= tf.sequence_mask(len_feas,
                              maxlen=tf.shape(outputs)[1],
                              dtype=tf.float32)[:, : , None]

        return outputs, len_feas


class CONV_2D(CONV_1D):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''
    def __init__(self, args, training, embed_table=None, name=None):
        self.num_layers = args.model.encoder.num_layers
        self.num_filters = args.model.encoder.num_filters
        super().__init__(args, training, embed_table, name)

    def encode(self, features, len_feas):
        size_batch = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        size_feat = features.get_shape()[-1]
        x = batch_splice(features, 1, 1, jump=False)
        x = tf.layers.max_pooling1d(x, 2, 2, 'same')
        size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
        len_feas = tf.cast(tf.ceil(tf.cast(len_feas, tf.float32)/2), tf.int32)
        x = tf.reshape(x, [size_batch, size_length, size_feat, 3])

        for i in range(self.num_layers):
            x = normal_conv(
                inputs=x,
                filter_num=self.num_filters,
                kernel=(3,9),
                stride=(2,1),
                padding='SAME',
                use_relu=True,
                name="conv_"+str(i),
                norm_type=None
                )
            size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
            len_feas = tf.cast(tf.ceil(tf.cast(len_feas, tf.float32)/2), tf.int32)

        outputs = tf.reshape(x, [size_batch, size_length, self.num_filters*size_feat])
        outputs *= tf.sequence_mask(len_feas,
                              maxlen=tf.shape(outputs)[1],
                              dtype=tf.float32)[:, : , None]

        return outputs, len_feas


class CONV_1D_with_RNN(CONV_1D):
    '''VERY DEEP CONVOLUTIONAL NETWORKS FOR END-TO-END SPEECH RECOGNITION
    '''
    def __init__(self, args, training, embed_table=None, name=None):
        self.num_blocks = args.model.encoder.num_blocks
        self.num_filters = args.model.encoder.num_filters
        self.rnn_hidden_size = args.model.encoder.rnn_hidden_size
        super().__init__(args, training, embed_table, name)

    def encode(self, features, len_feas):

        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]

        x = tf.layers.dense(features, units=self.num_filters, use_bias=True)
        len_x = len_feas
        for i in range(self.num_blocks):
            inputs = x
            x = tf.layers.conv1d(x, filters=self.num_filters, kernel_size=5, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(x, filters=self.num_filters, kernel_size=5, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = inputs + 0.3*x
            x = tf.layers.max_pooling1d(x, 2, 2, 'same')
            size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
            len_x = tf.cast(tf.ceil(tf.cast(len_x, tf.float32)/2), tf.int32)
        hidden1 = tf.reshape(x, [size_batch, size_length, self.num_filters])

        hidden2 = blstm(
            x=features,
            len_feas=len_feas,
            hidden_size=self.rnn_hidden_size,
            name='blstm')
        for i in range(self.num_blocks):
            hidden2 = tf.layers.max_pooling1d(hidden2, 2, 2, 'same')

        hidden = tf.concat([hidden1, hidden2], 2)
        hidden *= tf.sequence_mask(
            len_x,
            maxlen=tf.shape(hidden)[1],
            dtype=tf.float32)[:, : , None]

        return hidden, len_x
