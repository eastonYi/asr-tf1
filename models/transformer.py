'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only predsself.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

import tensorflow as tf
import logging

from .seq2seqModel import Seq2SeqModel


class Transformer(Seq2SeqModel):
    '''a general class for an encoder decoder system
    '''

    def __init__(self, tensor_global_step, encoder, decoder, training, args,\
                 batch=None, name='transformer'):
        '''Model constructor
        Args:
        '''
        self.name = name
        self.args = args
        self.training = training
        super().__init__(tensor_global_step, encoder, decoder, training,\
                         args, batch, name=name)

    def __call__(self, feature, len_features, labels=None, len_labels=None, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            encoder = self.gen_encoder(
                training=self.training,
                args=self.args)
            decoder = self.gen_decoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args)

            with tf.variable_scope(encoder.name or 'encoder'):
                encoded, len_encoded = encoder(feature, len_features)

            with tf.variable_scope(decoder.name or 'decoder'):
                if not self.training: # infer phrases
                    if self.args.beam_size>1:
                        logging.info('beam search with language model ...')
                        results, preds, len_decoded = decoder.beam_decode_rerank(
                            encoded,
                            len_encoded)
                    else:
                        logging.info('gready search ...')
                        logits, preds, len_decoded = decoder.decoder_with_caching(
                            encoded,
                            len_encoded)
                else:
                    logging.info('teacher-forcing training ...')
                    assert len_labels is not None
                    labels_sos = decoder.build_input(labels)

                    logits, preds, len_decoded = decoder(
                        encoded=encoded,
                        len_encoded=len_encoded,
                        decoder_input=labels_sos)

        return logits, preds, len_decoded
