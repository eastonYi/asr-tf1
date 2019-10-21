'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import tensorflow as tf
from .decoder import Decoder

class FCDecoder(Decoder):
    '''a fully connected decoder for the CTC architecture'''

    def decode(self, encoded, len_encoded, decoder_input, training):
        dim_output = self.args.dim_output
        logits = tf.layers.dense(
            inputs=encoded,
            units=dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')

        len_decoded = len_encoded
        logits *= tf.tile(tf.expand_dims(tf.sequence_mask(len_decoded, tf.shape(logits)[1], tf.float32), -1),
                           [1, 1, dim_output])
        decoded = tf.argmax(logits, -1)

        return logits, decoded, len_decoded
