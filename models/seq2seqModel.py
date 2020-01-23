'''@file model.py
contains de Model class
During the training , using greadysearch decoder, there is loss.
During the dev/infer, using beamsearch decoder, there is no logits, therefor loss, only preds self.
because we cannot access label during the dev set and need to depend on last time decision.

so, there is only infer and training
'''

import tensorflow as tf
import logging
from collections import namedtuple

from .lstmModel import LSTM_Model
from .utils.tools import choose_device, smoothing_cross_entropy


class Seq2SeqModel(LSTM_Model):
    '''a general class for an encoder decoder system
    '''

    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 batch=None, name='seq2seqModel'):
        '''Model constructor

        Args:
        '''
        self.name = name
        self.args = args
        self.training = training
        self.gen_encoder = encoder # encoder class
        self.gen_decoder = decoder # decoder class

        super().__init__(tensor_global_step, training, args, batch=batch, name=name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                training=self.training,
                args=self.args)
            decoder = self.gen_decoder(
                training=self.training,
                embed_table=self.embed_table,
                global_step=self.global_step,
                args=self.args)

            encoded, len_encoded = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_features=tensors_input.len_feat_splits[id_gpu])

            decoder_input = decoder.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)
            # if in the infer, the decoder_input.input_labels and len_labels are None
            logits, preds, len_decoded = decoder(encoded, len_encoded, decoder_input.input_labels)

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
            return loss, gradients, [len_decoded, preds, tensors_input.label_splits[id_gpu]]
        else:
            return logits, len_decoded, preds

    def build_infer_graph(self):
        tensors_input = self.build_infer_input()

        logits, len_logits, preds = self.build_single_graph(
            id_gpu=0,
            name_gpu=self.list_gpu_devices[0],
            tensors_input=tensors_input)

        if preds.get_shape().ndims == 3:
            preds = preds[:,:,0]

        return preds, tensors_input.shape_batch, len_logits

    def ce_loss(self, logits, labels, len_labels):
        """
        Compute optimization loss.
        batch major
        """
        with tf.name_scope('CE_loss'):
            crossent = smoothing_cross_entropy(
                logits=logits,
                labels=labels,
                vocab_size=self.args.dim_output,
                confidence=self.args.model.decoder.label_smoothing)

            mask = tf.sequence_mask(
                len_labels,
                maxlen=tf.shape(logits)[1],
                dtype=logits.dtype)
            loss = tf.reduce_sum(crossent * mask, -1)/tf.reduce_sum(mask, -1)

        return loss

    def build_infer_input(self):
        """
        used for inference. For inference must use placeholder.
        during the infer, we only get the decoded result and not use label
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_fea_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                self.list_pl = [batch_features, batch_fea_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.len_feat_splits = tf.split(batch_fea_lens, self.num_gpus, name="len_feat_splits")
                tensors_input.label_splits = None
                tensors_input.len_label_splits = None

        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input
