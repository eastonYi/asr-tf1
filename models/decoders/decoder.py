'''@file asr_decoder.py
contains the EDDecoder class'''

from abc import ABCMeta, abstractmethod
import tensorflow as tf
from collections import namedtuple


class Decoder(object):
    '''a general decoder for an encoder decoder system
    converts the high level features into output logits
    '''

    __metaclass__ = ABCMeta

    def __init__(self, args, training, global_step, name=None):
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
        self.training = training
        self.global_step = global_step
        self.dim_output = self.args.dim_output

    def teacher_forcing(self, encoded, len_encoded, target_labels, max_len):
        with tf.variable_scope(self.name or 'decoder'):
            logits = self.teacherforcing_decode(encoded, len_encoded, target_labels, max_len)
        return logits

    def max_decoder_len(self, len_src=None):
        if self.args.model.decoder.max_decoded_len:
            len_max_decode = self.args.model.decoder.max_decoded_len
        else:
            assert len_src
            decoding_length_factor = 1.0
            len_max_decode = tf.to_int32(tf.round(
                tf.to_float(len_src) * decoding_length_factor))

        return len_max_decode

    def embedding(self, ids, embed_table=None):
        if embed_table is not None:
            embeded = tf.nn.embedding_lookup(embed_table, ids)
        else:
            embeded = tf.one_hot(ids, self.dim_output, dtype=tf.float32)

        return embeded

    def gen_embedding(self, size_input, size_embedding):
        with tf.device("/cpu:0"):
            with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                embed_table = tf.get_variable(
                    "embedding", [size_input, size_embedding], dtype=tf.float32)

        return embed_table

    @property
    def variables(self):
        '''
        get a list of the models's variables
        '''
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '/')

        return variables
