'''@file asr_decoder.py
contains the EDDecoder class'''
import tensorflow as tf
from tensorflow.python.util import nest

from tfModels.layers import make_multi_cell
from tfSeq2SeqModels.tools.utils import dense


class LM_Decoder(object):
    '''a general decoder for an encoder decoder system
    converts the high level features into output logits
    '''
    def __init__(self, args, is_train, embed_table=None, name=None):
        '''EDDecoder constructor
        Args:
            conf: the decoder configuration as a configparser
            outputs: the name of the outputs of the model
            constraint: the constraint for the variables

            self.start_token is used in the infer_graph, for auto feed the first
            <sos> tokens to the decoder, while in the train_graph, you need to
            pad the <sos> for the decoder input manually!
            Also, in the infer_graph, decoder should know when to stop, so the
            decoder need to specify the <eos> in the helper or BeamSearchDecoder.
        '''
        self.args = args
        self.name = name
        self.is_train = is_train
        self.num_cell_units = args.model.decoder.num_cell_units
        self.dropout = args.model.decoder.dropout
        self.keep_prob = 1 - args.model.decoder.dropout
        self.cell_type = args.model.decoder.cell_type
        self.num_layers = args.model.decoder.num_layers
        self.init_scale = args.model.decoder.init_scale
        self.rnn_mode = args.model.decoder.rnn_mode
        self.size_embedding = args.model.decoder.size_embedding
        self.dim_output = args.dim_output
        self.embed_table = embed_table

        self.cell = make_multi_cell(
            num_cell_units=self.num_cell_units,
            is_train=self.is_train,
            keep_prob=1-self.dropout,
            rnn_mode='BLOCK',
            num_layers=self.num_layers)

    def __call__(self, inputs, len_inputs):
        '''
        Create the variables and do the forward computation to decode an entire
        sequence
        Inputs:
            inputs.set_shape([None, None, self.size_embedding])
        Returns:
            - the output logits of the decoder as a dictionary of
                [batch_size x time x ...] tensors
            - the logit sequence_lengths as a dictionary of [batch_size] vectors
            - the final state of the decoder as a possibly nested tupple
                of [batch_size x ... ] tensors
        '''
        with tf.variable_scope(self.name or 'decoder'):
            self.scope = tf.get_variable_scope()
            if self.is_train and self.keep_prob < 1.0:
                inputs = tf.nn.dropout(inputs, self.keep_prob)

            hidden_output, final_state = tf.nn.dynamic_rnn(
                cell=self.cell,
                inputs=inputs,
                sequence_length=len_inputs,
                dtype=tf.float32)

            self.fully_connected = tf.get_variable(
                'fully_connected',
                shape=[self.num_cell_units, self.dim_output])
            logits = dense(
                inputs=hidden_output,
                units=self.dim_output,
                kernel=tf.transpose(self.fully_connected),
                use_bias=False)

        return logits

    def embedding(self, ids):
        if self.embed_table:
            embeded = tf.nn.embedding_lookup(self.embed_table, ids)
        else:
            embeded = tf.one_hot(ids, self.args.dim_output, dtype=tf.float32)

        return embeded

    def zero_state(self, batch_size, dtype=tf.float32):
        return self.cell.zero_state(batch_size, dtype=tf.float32)

    def forward(self, input, state, stop_gradient=False, list_state=False):
        if input.get_shape().ndims <2:
            input = tf.nn.embedding_lookup(self.embed_table, input)
        hidden_output, state = tf.contrib.legacy_seq2seq.rnn_decoder(
            decoder_inputs=[input],
            initial_state=state,
            cell=self.cell)
        cur_logit = dense(
            inputs=hidden_output[0],
            units=self.dim_output,
            kernel=tf.transpose(self.fully_connected),
            use_bias=False)

        pred = tf.to_int32(tf.argmax(cur_logit, -1))

        if stop_gradient:
            cur_logit = tf.stop_gradient(cur_logit)

        if list_state:
            list_cells = []
            for cell in state:
                cell = tf.nn.rnn_cell.LSTMStateTuple(tf.stop_gradient(cell[0]), tf.stop_gradient(cell[1]))
                list_cells.append(cell)
            state = tuple(list_cells)

        prob = tf.nn.softmax(cur_logit)

        return prob, pred, state

    def score(self, decoder_input, len_seqs):
        '''
        decoder_input : <sos> + sent (+ <eos>)
        score batch sentences
        utilize the `decoder_impl`
        return batch_score(log scale)
        note: the pad score should be removed after the tf.log! otherwise the pad is 0 and
        tf.log(0.0) would pull down the sentence score!
        the `len_seqs` not counts the <sos> !!!
        '''

        decoder_input = tf.to_int32(decoder_input)
        # input is `<sos> + sent`
        decoder_input_embed = tf.nn.embedding_lookup(self.embed_table, decoder_input[:, :-1])
        hidden_output, final_state = tf.nn.dynamic_rnn(
            cell=self.cell,
            inputs=decoder_input_embed,
            sequence_length=len_seqs,
            dtype=tf.float32)
        # reuse the `fully_connected`
        logits = dense(
            inputs=hidden_output,
            units=self.dim_output,
            kernel=tf.transpose(self.fully_connected),
            use_bias=False)
        distribution = tf.nn.softmax(logits, -1)
        # output is `sent + <eos>`
        scores = tf.gather_nd(distribution, self.tensor2indices(decoder_input[:, 1:]))

        scores_log = tf.log(scores)
        mask = tf.sequence_mask(len_seqs, maxlen=tf.reduce_max(len_seqs), dtype=scores.dtype)
        scores_log = scores_log * mask
        scores_log = tf.reduce_sum(scores_log, -1)

        return scores_log, distribution

    def sample(self, token_init=None, state_init=None, max_length=50):
        def step(i, preds, state_decoder):
            output, state_output = self.forward(preds[:, -1], state_decoder)
            sampled_id = tf.distributions.Categorical(logits=output).sample()
            sampled_ids = tf.concat([preds, tf.expand_dims(sampled_id, 1)], axis=1)

            return i+1, sampled_ids, state_output

        num_samples = tf.placeholder(tf.int32, [], name='num_samples')

        if token_init is None:
            token_init = tf.ones([num_samples, 1], dtype=tf.int32) * self.args.sos_idx
        if state_init is None:
            state_init = self.zero_state(num_samples)

        _, sampled, _ = tf.while_loop(
            cond=lambda i, *_: tf.less(i, max_length),
            body=step,
            loop_vars=[0, token_init, state_init],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None]),
                              nest.map_structure(lambda t: tf.TensorShape(t.shape), state_init)
                              ]
            )
        return sampled, num_samples

    @staticmethod
    def tensor2indices(batch_sents):
        """
        batch_sents = tf.constant([[2,3,3,1,4],
                                   [4,3,5,1,0]], dtype=tf.int32)
        sess.run(tensor2indices(batch_sents))
        >>>
        array([[[0, 0, 2],
                [0, 1, 3],
                [0, 2, 3],
                [0, 3, 1],
                [0, 4, 4]],

               [[1, 0, 4],
                [1, 1, 3],
                [1, 2, 5],
                [1, 3, 1],
                [1, 4, 0]]], dtype=int32)
        """
        size_batch = tf.shape(batch_sents)[0]
        len_batch = tf.shape(batch_sents)[1]
        batch_i = tf.range(size_batch)
        len_i = tf.range(len_batch)

        # [0,0,0,1,1,1,2,2,2,...]
        batch_i = tf.tile(batch_i[:, None], [1, len_batch])
        # [0,1,2,0,1,2,0,1,2,...]
        len_i = tf.tile(len_i[None, :], [size_batch, 1])

        indices = tf.stack([batch_i, len_i, batch_sents], -1)

        return indices
