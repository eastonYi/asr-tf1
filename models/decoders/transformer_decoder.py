import tensorflow as tf

from .rna_decoder import RNADecoder
from utils.tfTools.attention import residual, multihead_attention, ff_hidden,\
    attention_bias_ignore_padding, add_timing_signal_1d, attention_bias_lower_triangle

inf = 1e10


class Transformer_Decoder(RNADecoder):

    def __init__(self, args, training, global_step, embed_table=None, name=None):
        super().__init__(args, training, global_step, embed_table, name)
        # use decoder heres
        self.num_blocks = args.model.decoder.num_blocks
        self.num_cell_units = args.model.decoder.num_cell_units
        self.attention_dropout_rate = args.model.decoder.attention_dropout_rate if training else 0.0
        self.residual_dropout_rate = args.model.decoder.residual_dropout_rate if training else 0.0
        self.num_heads = args.model.decoder.num_heads
        self.size_embedding = args.model.decoder.size_embedding
        self._ff_activation = (lambda x, y: x * tf.sigmoid(y)) \
                if args.model.decoder.activation == 'glu' else tf.nn.relu # glu
        self.softmax_temperature = args.model.decoder.softmax_temperature
        self.lambda_lm = self.args.lambda_lm

    def decode(self, encoded, len_encoded, decoder_input):
        """
        used for MLE training
        """
        decoder_output = self.decoder_impl(decoder_input, encoded, len_encoded)

        logits = tf.layers.dense(
            decoder_output,
            self.args.dim_output,
            use_bias=False,
            name='decoder_fc')

        preds = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, preds

    def decoder_with_caching(self, encoded, len_encoded):
        """
        gread search, used for self-learning training or infer
        """
        batch_size = tf.shape(encoded)[0]
        token_init = tf.fill([batch_size, 1], self.start_token)
        logits_init = tf.zeros([batch_size, 1, self.dim_output], dtype=tf.float32)
        finished_init = tf.zeros([batch_size], dtype=tf.bool)
        len_decoded_init = tf.ones([batch_size], dtype=tf.int32)
        cache_decoder_init = tf.zeros([batch_size, 0, self.num_blocks, self.num_cell_units])
        encoder_padding = tf.equal(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1]), False) # bool tensor
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        def step(i, preds, cache_decoder, logits, len_decoded, finished):

            preds_emb = self.embedding(preds)
            decoder_input = preds_emb

            decoder_output, cache_decoder = self.decoder_with_caching_impl(
                decoder_input,
                cache_decoder,
                encoded,
                encoder_attention_bias)

            cur_logit = tf.layers.dense(
                inputs=decoder_output[:, -1, :],
                units=self.dim_output,
                activation=None,
                use_bias=False,
                name='decoder_fc')

            cur_ids = tf.to_int32(tf.argmax(cur_logit, -1))
            preds = tf.concat([preds, cur_ids[:, None]], axis=1)
            logits = tf.concat([logits, cur_logit[:, None]], 1)

            # Whether sequences finished.
            has_eos = tf.equal(cur_ids, self.end_token)
            finished = tf.logical_or(finished, has_eos)
            len_decoded += 1-tf.to_int32(finished)

            return i+1, preds, cache_decoder, logits, len_decoded, finished

        def not_finished(i, preds, cache, logit, len_decoded, finished):
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(finished)),
                tf.less(
                    i,
                    tf.reduce_min([tf.shape(encoded)[1], self.args.max_len]) # maxlen = 25
                )
            )

        i, preds, cache_decoder, logits, len_decoded, finished = tf.while_loop(
            cond=not_finished,
            body=step,
            loop_vars=[0, token_init, cache_decoder_init, logits_init, len_decoded_init, finished_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None, None, None, None]),
                              tf.TensorShape([None, None, self.dim_output]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None])]
            )
        # len_decoded = tf.Print(len_decoded, [finished], message='finished: ', summarize=1000)
        len_decoded -= 1-tf.to_int32(finished) # for decoded length cut by encoded length
        logits = logits[:, 1:, :]
        preds = preds[:, 1:]
        not_padding = tf.sequence_mask(len_decoded, dtype=tf.int32)
        preds = tf.multiply(tf.to_int32(preds), not_padding)

        return logits, preds, len_decoded

    def decoder_with_caching_impl(self, decoder_input, decoder_cache, encoder_output, encoder_attention_bias):
        # Positional Encoding
        decoder_input += add_timing_signal_1d(decoder_input)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_input,
                                           rate=self.residual_dropout_rate,
                                           training=self.training)
        new_cache = []

        # rest block with residual
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                # the caching_impl only need to calculate decoder_output[:, -1:, :] !
                decoder_output = residual(decoder_output[:, -1:, :],
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=None,
                                              total_key_depth=self.num_cell_units,
                                              total_value_depth=self.num_cell_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              num_queries=1,
                                              output_depth=self.num_cell_units,
                                              name="decoder_self_attention",
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self.num_cell_units,
                                              total_value_depth=self.num_cell_units,
                                              output_depth=self.num_cell_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              num_queries=1,
                                              name="decoder_vanilla_attention",
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self.num_cell_units,
                                              output_size=self.num_cell_units,
                                              activation=self._ff_activation),
                                          dropout_rate=self.residual_dropout_rate)

                decoder_output = tf.concat([decoder_cache[:, :, i, :], decoder_output], axis=1)
                new_cache.append(decoder_output[:, :, None, :])

        new_cache = tf.concat(new_cache, axis=2)  # [batch_size, n_step, num_blocks, num_hidden]

        return decoder_output, new_cache

    def decoder_impl(self, decoder_input, encoder_output, len_encoded):
        # encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_padding = tf.equal(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoder_output)[1]), False) # bool tensor
        # [-0 -0 -0 -0 -0 -0 -0 -0 -0 -1e+09] the pading place is -1e+09
        encoder_attention_bias =attention_bias_ignore_padding(encoder_padding)

        decoder_output = self.embedding(decoder_input)
        # Positional Encoding
        decoder_output +=add_timing_signal_1d(decoder_output)
        # Dropout
        decoder_output = tf.layers.dropout(decoder_output,
                                           rate=self.residual_dropout_rate,
                                           training=self.training)
        # Bias for preventing peeping later information
        self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention (self-attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=None,
                                              bias=self_attention_bias,
                                              total_key_depth=self.num_cell_units,
                                              total_value_depth=self.num_cell_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              output_depth=self.num_cell_units,
                                              name="decoder_self_attention",
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Multihead Attention (vanilla attention)
                decoder_output = residual(decoder_output,
                                          multihead_attention(
                                              query_antecedent=decoder_output,
                                              memory_antecedent=encoder_output,
                                              bias=encoder_attention_bias,
                                              # bias=None,
                                              total_key_depth=self.num_cell_units,
                                              total_value_depth=self.num_cell_units,
                                              output_depth=self.num_cell_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              name="decoder_vanilla_attention",
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Feed Forward
                decoder_output = residual(decoder_output,
                                          ff_hidden(
                                              decoder_output,
                                              hidden_size=4 * self.num_cell_units,
                                              output_size=self.num_cell_units,
                                              activation=self._ff_activation),
                                          dropout_rate=self.residual_dropout_rate)

        return decoder_output

    def forward(self, i, preds, state_decoder):
        """
        self.cell
        self.encoded
        """
        prev_emb = self.embedding(preds[:, -1])
        decoder_input = tf.concat([self.encoded[:, i, :], prev_emb], axis=1)
        decoder_input.set_shape([None, self.num_cell_units_en+self.size_embedding])

        with tf.variable_scope(self.name or 'decoder', reuse=True):
            with tf.variable_scope("decoder_lstms"):
                output_decoder, state_decoder = tf.contrib.legacy_seq2seq.rnn_decoder(
                    decoder_inputs=[decoder_input],
                    initial_state=state_decoder,
                    cell=self.cell)

            cur_logit = tf.layers.dense(
                inputs=output_decoder[0],
                units=self.dim_output,
                activation=None,
                use_bias=False,
                name='fully_connected'
                )
            cur_ids = tf.to_int32(tf.argmax(cur_logit, -1))

        return cur_ids, state_decoder
