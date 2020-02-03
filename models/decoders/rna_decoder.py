'''@file rnn_decoder.py
the while_loop implementation'''

import tensorflow as tf
from .decoder import Decoder
from ..utils.blocks import make_multi_cell
from tensorflow.python.util import nest

inf = 1e10


class RNADecoder(Decoder):
    """language model cold fusion
    """

    def __init__(self, args, training, global_step, name='RNADecoder'):
        self.num_layers = args.model.decoder.num_layers
        self.num_cell_units_de = args.model.decoder.hidden_size
        self.num_cell_units_en = args.model.encoder.hidden_size
        self.size_embedding = args.model.decoder.size_embedding
        self.dropout = args.model.dropout
        self.dim_output = args.dim_output
        self.embed_table = self.gen_embedding(self.dim_output, self.size_embedding)
        super().__init__(args, training, global_step, name)

    def __call__(self, encoded, len_encoded, decoder_input):
        batch_size = tf.shape(len_encoded)[0]
        blank_id = self.dim_output-1
        token_init = tf.fill([batch_size, 1], blank_id)
        logits_init = tf.zeros([batch_size, 1, self.dim_output], dtype=tf.float32)

        self.cell = self.create_cell()
        # collect the initial states of lstms used in decoder.
        all_initial_states = {}
        all_initial_states["state_decoder"] = self.zero_state(batch_size, dtype=tf.float32)

        def step(i, preds, all_states, logits):
            state_decoder = all_states["state_decoder"]
            prev_emb = self.embedding(preds[:, -1], self.embed_table)
            decoder_input = tf.concat([encoded[:, i, :], prev_emb], axis=1)
            decoder_input.set_shape([None, self.size_embedding + self.num_cell_units_en])

            # Lstm part
            with tf.variable_scope("decoder_lstms"):
                output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_input],
                    initial_state=state_decoder,
                    cell=self.cell)
                all_states["state_decoder"] = state_decoder
                output_decoder = [tf.concat([output_decoder[0], encoded[:, i, :]], axis=1)]

                cur_logit = tf.layers.dense(
                    inputs=output_decoder[0],
                    units=self.dim_output,
                    activation=None,
                    use_bias=False,
                    name='fully_connected'
                    )

            if self.training and self.args.model.decoder.sample_decoder:
                cur_ids = tf.distributions.Categorical(logits=cur_logit/self.softmax_temperature).sample()
            else:
                cur_ids = tf.to_int32(tf.argmax(cur_logit, -1))
            preds = tf.concat([preds, tf.expand_dims(cur_ids, 1)], axis=1)
            logits = tf.concat([logits, tf.expand_dims(cur_logit, 1)], 1)

            return i+1, preds, all_states, logits

        _, preds, _, logits = tf.while_loop(
            cond=lambda i, *_: tf.less(i, tf.shape(encoded)[1]),
            body=step,
            loop_vars=[0, token_init, all_initial_states, logits_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), all_initial_states),
                              tf.TensorShape([None, None, self.dim_output])]
            )

        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.to_int32(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1]))
        preds = tf.multiply(tf.to_int32(preds), not_padding)

        return logits, preds, len_encoded

    def create_cell(self):
        cell = make_multi_cell(
            self.num_cell_units_de,
            self.training,
            1-self.dropout,
            self.num_layers,
            rnn_mode='BLOCK')

        return cell

    def zero_state(self, batch_size, dtype):
        return self.cell.zero_state(batch_size, dtype)
