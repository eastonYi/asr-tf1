import tensorflow as tf

from .rna_decoder import RNADecoder
from ..utils.blocks import dense_without_vars
from ..utils.attention import residual, multihead_attention, ff_hidden,\
    attention_bias_ignore_padding, add_timing_signal_1d, attention_bias_lower_triangle

inf = 1e10


class Transformer_Decoder(RNADecoder):

    def __init__(self, args, training, global_step, name='decoder'):
        self.name = name
        self.args = args
        self.num_blocks = args.model.decoder.num_blocks
        self.num_cell_units = args.model.decoder.num_cell_units
        self.max_decode_len = args.model.decoder.max_decode_len
        self.attention_dropout_rate = args.model.decoder.attention_dropout_rate if training else 0.0
        self.residual_dropout_rate = args.model.decoder.residual_dropout_rate if training else 0.0
        self.num_heads = args.model.decoder.num_heads
        self.size_embedding = args.model.decoder.size_embedding
        self._ff_activation = lambda x, y: x * tf.sigmoid(y)
        self.lambda_lm = self.args.lambda_lm
        super().__init__(args, training, global_step, name)

    def decode(self, encoded, len_encoded, decoder_input):
        # used for MLE training
        decoder_output = self.decoder_impl(encoded, len_encoded, decoder_input)
        logits = tf.layers.dense(
            decoder_output,
            self.args.dim_output,
            use_bias=False,
            name='decoder_fc')

        preds = tf.to_int32(tf.argmax(logits, axis=-1))

        return logits, preds, tf.no_op()

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
                    tf.reduce_min([tf.shape(encoded)[1], self.max_decode_len]) # maxlen = 25
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

    def decoder_impl(self, encoder_output, len_encoded, decoder_input):
        # encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_padding = tf.equal(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoder_output)[1]), False) # bool tensor
        # [-0 -0 -0 -0 -0 -0 -0 -0 -0 -1e+09] the pading place is -1e+09
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        decoder_output = self.embedding(decoder_input)
        # Positional Encoding
        decoder_output += add_timing_signal_1d(decoder_output)
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

    def beam_decode_rerank(self, encoded, len_encoded):
        """
        beam search rerank at end with language model integration (self-attention model)
        the input to te score is <sos> + tokens !!!
        """
        beam_size = self.beam_size
        batch_size = tf.shape(len_encoded)[0]

        # beam search Initialize
        # repeat each sample in batch along the batch axis [1,2,3,4] -> [1,1,2,2,3,3,4,4]
        encoded = tf.tile(encoded[:, None, :, :],
                          multiples=[1, beam_size, 1, 1]) # [batch_size, beam_size, *, hidden_units]
        encoded = tf.reshape(encoded,
                             [batch_size * beam_size, -1, encoded.get_shape()[-1].value])
        len_encoded = tf.reshape(tf.tile(len_encoded[:, None], multiples=[1, beam_size]), [-1]) # [batch_size * beam_size]

        # [[<S>, <S>, ..., <S>]], shape: [batch_size * beam_size, 1]
        token_init = tf.fill([batch_size * beam_size, 1], self.args.sos_idx)
        logits_init = tf.zeros([batch_size * beam_size, 0, self.dim_output], dtype=tf.float32)
        len_decoded_init = tf.ones_like(len_encoded, dtype=tf.int32)
        # the score must be [0, -inf, -inf, ...] at init, for the preds in beam is same in init!!!
        scores_init = tf.constant([0.0] + [-inf] * (beam_size - 1), dtype=tf.float32)  # [beam_size]
        scores_init = tf.tile(scores_init, multiples=[batch_size])  # [batch_size * beam_size]
        finished_init = tf.zeros_like(scores_init, dtype=tf.bool)

        cache_decoder_init = tf.zeros([batch_size*beam_size,
                                       0,
                                       self.num_blocks,
                                       self.num_cell_units])
        if self.lm:
            cache_lm_init = tf.zeros([batch_size*beam_size,
                                      0,
                                      self.lm.args.model.decoder.num_blocks,
                                      self.lm.args.model.decoder.num_cell_units])
        else:
            cache_lm_init = tf.zeros([0, 0, 0, 0])

        # collect the initial states of lstms used in decoder.
        base_indices = tf.reshape(tf.tile(tf.range(batch_size)[:, None], multiples=[1, beam_size]), shape=[-1])

        encoder_padding = tf.equal(tf.sequence_mask(len_encoded, maxlen=tf.shape(encoded)[1]), False) # bool tensor
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        def step(i, preds, scores, cache_decoder, cache_lm, logits, len_decoded, finished):
            """
            the cache has no specific shape, so no can be put in the all_states
            """
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

            logits = tf.concat([logits, cur_logit[:, None]], 1)
            z = tf.nn.log_softmax(cur_logit) # [batch*beam, size_output]

            # the langueage model infer
            if self.args.model.shallow_fusion:
                assert self.lm
                preds_emb = self.lm.decoder.embedding(preds)

                with tf.variable_scope(self.args.top_scope, reuse=True):
                    with tf.variable_scope(self.args.lm_scope):
                        lm_output, cache_lm = self.lm.decoder.decoder_with_caching_impl(preds_emb, cache_lm)
                        logit_lm = dense_without_vars(
                            inputs=lm_output[:, -1, :],
                            units=self.dim_output,
                            kernel=tf.transpose(self.lm.decoder.fully_connected),
                            use_bias=False)
                z_lm = self.lambda_lm * tf.nn.log_softmax(logit_lm) # [batch*beam, size_output]
            else:
                z_lm = tf.zeros_like(z)

            # rank the combined scores
            next_scores, next_preds = tf.nn.top_k(z+z_lm, k=beam_size, sorted=True)
            next_preds = tf.to_int32(next_preds)

            # beamed scores & Pruning
            scores = scores[:, None] + next_scores  # [batch_size * beam_size, beam_size]
            scores = tf.reshape(scores, shape=[batch_size, beam_size * beam_size])

            _, k_indices = tf.nn.top_k(scores, k=beam_size)
            k_indices = base_indices * beam_size * beam_size + tf.reshape(k_indices, shape=[-1])  # [batch_size * beam_size]
            # Update scores.
            scores = tf.reshape(scores, [-1])
            scores = tf.gather(scores, k_indices)
            # Update predictions.
            next_preds = tf.reshape(next_preds, shape=[-1])
            next_preds = tf.gather(next_preds, indices=k_indices)

            # k_indices: [0~batch*beam*beam], preds: [0~batch*beam]
            # preds, cache_lm, cache_decoder: these data are shared during the beam expand among vocab
            preds = tf.gather(preds, indices=k_indices // beam_size)
            cache_lm = tf.gather(cache_lm, indices=k_indices // beam_size)
            cache_decoder = tf.gather(cache_decoder, indices=k_indices // beam_size)
            preds = tf.concat([preds, next_preds[:, None]], axis=1)  # [batch_size * beam_size, i]

            has_eos = tf.equal(next_preds, self.end_token)
            finished = tf.logical_or(finished, has_eos)
            len_decoded += 1-tf.to_int32(finished)
            # i = tf.Print(i, [i], message='i: ', summarize=1000)

            return i+1, preds, scores, cache_decoder, cache_lm, logits, len_decoded, finished

        def not_finished(i, preds, scores, cache_decoder, cache_lm, logit, len_decoded, finished):
            # i = tf.Print(i, [i], message='i: ', summarize=1000)
            return tf.logical_and(
                tf.reduce_any(tf.logical_not(finished)),
                tf.less(
                    i,
                    tf.reduce_min([tf.shape(encoded)[1], self.args.max_len]) # maxlen = 100
                )
            )

        _, preds, scores_am, _, _, logits, len_decoded, finished = tf.while_loop(
            cond=not_finished,
            body=step,
            loop_vars=[0, token_init, scores_init, cache_decoder_init, cache_lm_init, logits_init, len_decoded_init, finished_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None, None, None, None]),
                              tf.TensorShape([None, None, None, None]),
                              tf.TensorShape([None, None, self.dim_output]),
                              tf.TensorShape([None]),
                              tf.TensorShape([None])]
            )

        # [batch_size * beam_size, ...]
        len_decoded -= 1-tf.to_int32(finished) # for decoded length cut by encoded length
        preds = preds[:, 1:]
        not_padding = tf.sequence_mask(len_decoded, dtype=tf.int32)
        preds *= not_padding

        # [batch_size , beam_size, ...]
        if self.args.model.rerank:
            assert self.lm
            with tf.variable_scope(self.args.top_scope, reuse=True):
                with tf.variable_scope(self.args.lm_scope):
                    scores_lm, distribution = self.lm.decoder.score(preds, len_decoded)

            scores_lm = self.args.lambda_rerank * scores_lm
        else:
            scores_lm = tf.zeros_like(scores_am)

        scores = scores_am + scores_lm

        # tf.nn.top_k is used to sort `scores`
        scores_sorted, sorted = tf.nn.top_k(tf.reshape(scores, [batch_size, beam_size]),
                                            k=beam_size,
                                            sorted=True)

        sorted = base_indices * beam_size + tf.reshape(sorted, shape=[-1])  # [batch_size * beam_size]

        # [batch_size * beam_size, ...]
        logits_sorted = tf.gather(logits, sorted)
        preds_sorted = tf.gather(preds, sorted)
        len_decoded_sorted = tf.gather(len_decoded, sorted)
        scores_lm_sorted = tf.gather(scores_lm, sorted)
        scores_am_sorted = tf.gather(scores_am, sorted)

        # [batch_size, beam_size, ...]
        scores_lm_sorted = tf.reshape(scores_lm_sorted, shape=[batch_size, beam_size])
        scores_am_sorted = tf.reshape(scores_am_sorted, shape=[batch_size, beam_size])
        preds_sorted = tf.reshape(preds_sorted, shape=[batch_size, beam_size, -1])  # [batch_size, beam_size, max_length]
        logits_sorted = tf.reshape(logits_sorted, [batch_size, beam_size, -1, self.dim_output])
        len_decoded_sorted = tf.reshape(len_decoded_sorted, [batch_size, beam_size])

        # return logits, final_preds, len_encoded
        return [logits_sorted, preds_sorted, len_decoded_sorted, scores_am_sorted, scores_lm_sorted], preds_sorted[:, 0, :], len_decoded_sorted[:, 0]

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
