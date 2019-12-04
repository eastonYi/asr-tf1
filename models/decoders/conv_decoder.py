'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import tensorflow as tf
from .decoder import Decoder


class CONV_Decoder(Decoder):
    '''a fully connected decoder for the CTC architecture'''

    def __call__(self, encoded, len_encoded, decoder_input):
        num_filters = self.args.model.decoder.num_filters
        hidden_size = self.args.model.decoder.hidden_size
        num_blocks = self.args.model.decoder.num_blocks
        dropout = self.args.model.dropout
        num_fc = self.args.model.decoder.num_fc
        dim_output = self.args.dim_output

        encoded *= tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1], dtype=tf.float32)[:, :, None]
        # x = encoded
        x = tf.layers.dense(encoded, num_filters, use_bias=False)

        for i in range(num_blocks):
            inputs = x
            x = tf.layers.conv1d(x, filters=num_filters, kernel_size=3, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(x, filters=num_filters, kernel_size=3, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = inputs + 0.3*x

        # x = tf.reshape(x, [-1, self.max_input_len, self.num_filters])
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
