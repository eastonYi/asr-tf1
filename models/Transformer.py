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
from .utils.tools import choose_device


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
        super().__init__(tensor_global_step, encoder, decoder, training, \
                         args, batch, name=name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                training=self.training,
                args=self.args)
            decoder = self.gen_decoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args)

            with tf.variable_scope(encoder.name or 'encoder'):
                encoded, len_encoded = encoder(
                    features=tensors_input.feature_splits[id_gpu],
                    len_features=tensors_input.len_feat_splits[id_gpu])

            with tf.variable_scope(decoder.name or 'decoder'):
                decoder_input = decoder.build_input(
                    id_gpu=id_gpu,
                    tensors_input=tensors_input)

                if (not self.training) or (self.args.model.training_type == 'self-learning'):
                    '''
                    training_type:
                        - self-learning: get logits fully depend on self
                        - teacher-forcing: get logits depend on labels during training
                    '''
                    # infer phrases
                    if self.args.beam_size>1:
                        logging.info('beam search with language model ...')
                        results, preds, len_decoded = decoder.beam_decode_rerank(
                            encoded,
                            len_encoded)
                    else:
                        logging.info('gready search ...')
                        results, preds, len_decoded = decoder.decoder_with_caching(
                            encoded,
                            len_encoded)
                else:
                    logging.info('teacher-forcing training ...')
                    decoder_input_labels = decoder_input.input_labels * tf.sequence_mask(
                        decoder_input.len_labels,
                        maxlen=tf.shape(decoder_input.input_labels)[1],
                        dtype=tf.int32)
                    logits, preds, _ = decoder.decode(
                        encoded=encoded,
                        len_encoded=len_encoded,
                        decoder_input=decoder_input_labels)

            if self.training:
                loss = self.ce_loss(
                    logits=logits,
                    labels=decoder_input.output_labels[:, :tf.shape(logits)[1]],
                    len_labels=decoder_input.len_labels)

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.training:
            # no_op is preserved for debug info to pass
            return loss, gradients, [preds, tensors_input.label_splits[id_gpu]]
        else:
            return results, len_decoded, preds
