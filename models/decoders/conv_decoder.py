'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import tensorflow as tf
from .decoder import Decoder
from ..utils.blocks import normal_conv, block, blstm
from ..utils.tfAudioTools import batch_splice

class CONV_Decoder(Decoder):
    '''a fully connected decoder for the CTC architecture'''

    def __call__(self, encoded, len_encoded, decoder_input):
        num_filters = 64
        hidden_size = self.args.model.decoder.hidden_size
        num_blocks = self.args.model.decoder.num_blocks
        dropout = self.args.model.dropout
        num_fc = self.args.model.decoder.num_fc
        num_featmaps = self.args.model.decoder.left_splice + self.args.model.decoder.right_splice + 1
        dim_output = self.args.dim_output
        left_splice = self.args.model.decoder.left_splice
        right_splice = self.args.model.decoder.right_splice
        encoded *= tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1], dtype=tf.float32)[:, :, None]
        x = encoded
        size_feat = int(int(encoded.get_shape()[-1])/num_featmaps)
        size_batch  = tf.shape(encoded)[0]
        size_length = tf.shape(encoded)[1]
        # x = tf.reshape(encoded, [size_batch*size_length, 4, size_feat])
        encoded = batch_splice(encoded, left_splice, right_splice, jump=False)
        x = tf.reshape(encoded, [size_batch, size_length, size_feat, num_featmaps])
        x = normal_conv(
            inputs=x,
            filter_num=num_filters,
            kernel=(5,9),
            stride=(1,1),
            padding='SAME',
            use_relu=True,
            name="conv_1",
            norm_type=None
            )
        x = tf.reshape(x, [size_batch, size_length, size_feat*num_filters])
        if self.args.model.decoder.half:
            x = tf.layers.max_pooling1d(x, 2, 2, 'same')
            size_length = tf.cast(tf.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
            len_encoded = tf.cast(tf.ceil(tf.cast(len_encoded, tf.float32)/2), tf.int32)
        # x = tf.reshape(encoded, [size_batch, size_length, 11*size_feat])

        # x = tf.layers.dense(encoded, num_filters, use_bias=False)
        # for i in range(2):
        #     x = blstm(x, tf.ones([size_batch*size_length], tf.int32)*11, hidden_size, 'blstm_'+str(i))
        # x = tf.reshape(x[:, -1, :], [size_batch, size_length, hidden_size])

        # for i in range(num_blocks):
        #     inputs = x
        #     x = tf.layers.conv1d(x, filters=num_filters, kernel_size=3, strides=1, padding='same')
        #     x = tf.nn.relu(x)
        #     x = tf.layers.conv1d(x, filters=num_filters, kernel_size=3, strides=1, padding='same')
        #     x = tf.nn.relu(x)
        #     x = inputs + 0.3*x
        for i in range(num_fc):
            x = tf.layers.dense(x, units=hidden_size, use_bias=True)
            x = tf.nn.relu(x)
            x = tf.layers.dropout(x, rate=dropout, training=self.training)

        logits = tf.layers.dense(
            inputs=x,
            units=dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')
        logits *= tf.tile(tf.expand_dims(tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.float32), -1),
                          [1, 1, dim_output])
        preds = tf.argmax(logits, -1)

        return logits, preds, len_encoded
