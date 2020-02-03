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
from .utils.tools import choose_device, get_tensor_len


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
                if labels is None: # infer phrases
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

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):
        """
        It worth moting that tensors_input.len_label_splits need to add 1 along with the sos padding
        """
        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]
        labels = tensors_input.label_splits[id_gpu] if tensors_input.label_splits else None
        len_labels = get_tensor_len(labels) if tensors_input.len_label_splits else None

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):

            logits, preds, len_decoded = self(
                feature,
                len_features,
                labels,
                len_labels,
                reuse=reuse)

            if self.training:
                loss = self.ce_loss(
                    logits=logits,
                    labels=labels[:, :tf.shape(logits)[1]],
                    len_labels=len_labels)

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss, var_list=self.trainable_variables())

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.training:
            # no_op is preserved for debug info to pass
            return loss, gradients, [preds, tensors_input.label_splits[id_gpu]]
        else:
            return logits, len_decoded, preds
