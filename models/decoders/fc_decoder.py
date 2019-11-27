'''@file rnn_decoder.py
contains the general recurrent decoder class'''

import tensorflow as tf
from .decoder import Decoder


class FCDecoder(Decoder):
    '''a fully connected decoder for the CTC architecture'''

    def decode(self, encoded, len_encoded, decoder_input, shrink=False):
        dim_output = self.args.dim_output
        logits = tf.layers.dense(
            inputs=encoded,
            units=dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')

        if not shrink:
            len_logits = len_encoded
            logits *= tf.tile(tf.expand_dims(tf.sequence_mask(len_logits, tf.shape(logits)[1], tf.float32), -1),
                              [1, 1, dim_output])
            align = tf.argmax(logits, -1)

            return logits, align, len_logits
        else:
            batch_size = tf.shape(logits)[0]
            blank_id = tf.convert_to_tensor(dim_output - 1, dtype=tf.int64)
            frames_mark = tf.not_equal(tf.argmax(logits, -1), blank_id)
            prev = tf.concat([tf.ones([batch_size, 1], tf.int64) * blank_id, tf.argmax(logits, -1)[:, :-1]], 1)
            flag_norepeat = tf.not_equal(prev, tf.argmax(logits, -1))
            flag = tf.logical_and(flag_norepeat, frames_mark)
            flag = tf.logical_and(flag, tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.bool))
            len_labels = tf.reduce_sum(tf.cast(flag, tf.int32), -1)
            max_label_len = tf.reduce_max(len_labels)
            logits_output = tf.zeros([0, max_label_len, dim_output], tf.float32)

            def sent(b, logits_output):
                logit = tf.gather(logits[b, :, :], tf.where(flag[b, :])[:, 0])
                pad_logit = tf.zeros([tf.reduce_max([max_label_len - len_labels[b], 0]), dim_output])
                logits_padded = tf.concat([logit, pad_logit], 0)[:max_label_len, :]
                logits_output = tf.concat([logits_output, logits_padded[None, :]], 0)

                return b+1, logits_output

            _, logits_output = tf.while_loop(
            cond=lambda b, *_: tf.less(b, batch_size),
            body=sent,
            loop_vars=[0, logits_output],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None, dim_output])])

            len_decoded = len_labels
            preds = tf.argmax(logits_output, -1)

            return logits_output, preds, len_decoded
