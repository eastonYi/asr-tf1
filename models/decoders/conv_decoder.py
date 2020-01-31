'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import tensorflow as tf
from .decoder import Decoder
from ..utils.blocks import normal_conv, block, blstm
from ..utils.tfAudioTools import batch_splice


class CONV_Decoder(Decoder):
    '''a fully connected decoder for the CTC architecture'''
    def __init__(self, args, training, global_step, name=None):
        self.num_filters = args.model.decoder.num_filters
        self.hidden_size = args.model.decoder.hidden_size
        self.dropout = args.model.dropout
        self.num_fc = args.model.decoder.num_fc

        self.left_splice = args.model.decoder.left_splice
        self.right_splice = args.model.decoder.right_splice
        self.num_featmaps = self.left_splice + self.right_splice + 1
        self.dim_output = args.dim_output

        super().__init__(args, training, global_step, name)

    def decode(self, encoded, len_encoded, decoder_input):
        encoded *= tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1], dtype=tf.float32)[:, :, None]
        size_feat = int(encoded.get_shape()[-1])
        size_batch  = tf.shape(encoded)[0]
        size_length = tf.shape(encoded)[1]
        encoded = batch_splice(encoded, self.left_splice, self.right_splice, jump=False)
        x = tf.reshape(encoded, [size_batch, size_length, size_feat, self.num_featmaps])
        x = normal_conv(
            inputs=x,
            filter_num=self.num_filters,
            kernel=(5,9),
            stride=(1,1),
            padding='SAME',
            use_relu=True,
            name="decoder_conv",
            norm_type=None)
        x = tf.reshape(x, [size_batch, size_length, size_feat*self.num_filters])

        if self.args.model.decoder.half:
            x = tf.layers.max_pooling1d(x, 2, 2, 'same')
            len_encoded = tf.cast(tf.ceil(tf.cast(len_encoded, tf.float32)/2), tf.int32)

        for i in range(self.num_fc):
            x = tf.layers.dense(x, units=self.hidden_size, use_bias=True)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        logits = tf.layers.dense(
            inputs=x,
            units=self.dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')
        logits *= tf.tile(tf.expand_dims(tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.float32), -1),
                          [1, 1, self.dim_output])
        preds = tf.argmax(logits, -1)

        return logits, preds, len_encoded


class ResConv_Decoder(Decoder):
    '''a fully connected decoder for the CTC architecture'''
    def __init__(self, args, training, global_step, name=None):
        self.hidden_size = args.model.decoder.hidden_size
        self.num_filters = args.model.decoder.num_filters
        self.num_blocks = args.model.decoder.num_blocks
        self.dropout = args.model.dropout
        self.num_fc = args.model.decoder.num_fc
        self.dim_output = args.dim_output

        super().__init__(args, training, global_step, name)

    def decode(self, encoded, len_encoded, decoder_input):

        encoded *= tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1], dtype=tf.float32)[:, :, None]
        size_length = tf.shape(encoded)[1]

        x = tf.layers.dense(encoded, self.num_filters, use_bias=False)
        for i in range(self.num_blocks):
            inputs = x
            x = tf.layers.conv1d(x, filters=self.num_filters, kernel_size=3, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(x, filters=self.num_filters, kernel_size=1, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = inputs + 0.3*x

        if self.args.model.decoder.half:
            x = tf.layers.max_pooling1d(x, 2, 2, 'same')
            size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
            len_encoded = tf.cast(tf.ceil(tf.cast(len_encoded, tf.float32)/2), tf.int32)
#             x = x[:, ::2, :]
#             size_length = tf.cast(tf.floor(tf.cast(size_length, tf.float32)/2), tf.int32)
#             len_encoded = tf.cast(tf.floor(tf.cast(len_encoded, tf.float32)/2), tf.int32)

        for i in range(self.num_fc):
            x = tf.layers.dense(x, units=self.hidden_size, use_bias=True)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        logits = tf.layers.dense(
            inputs=x,
            units=self.dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')
        logits *= tf.tile(tf.expand_dims(tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.float32), -1),
                          [1, 1, self.dim_output])
        preds = tf.argmax(logits, -1)

        return logits, preds, len_encoded
