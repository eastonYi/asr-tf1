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
                 batch=None, embed_table_encoder=None, embed_table_decoder=None,
                 name='seq2seqModel'):
        '''Model constructor

        Args:
        '''
        self.name = name
        self.gen_encoder = encoder # encoder class
        self.gen_decoder = decoder # decoder class
        self.embed_table_encoder = self.get_embedding(
            embed_table=embed_table_encoder,
            size_input=args.model.encoder.size_vocab,
            size_embedding=args.model.encoder.size_embedding)
        self.embed_table_decoder = self.get_embedding(
            embed_table=embed_table_decoder,
            size_input=args.dim_output,
            size_embedding=args.model.decoder.size_embedding)
        if embed_table_encoder or (not encoder):
            """
            embed_table_encoder: MT
            no encoder, only decoder: LM
            """
            self.build_pl_input = self.build_idx_input
            self.build_infer_input = self.build_infer_idx_input

        super().__init__(tensor_global_step, training, args, batch=batch, name=name)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            encoder = self.gen_encoder(
                training=self.training,
                embed_table=self.embed_table_encoder,
                args=self.args)
            decoder = self.gen_decoder(
                training=self.training,
                embed_table=self.embed_table_decoder,
                global_step=self.global_step,
                args=self.args)
            self.schedule = decoder.schedule

            encoded, len_encoded = encoder(
                features=tensors_input.feature_splits[id_gpu],
                len_feas=tensors_input.len_feat_splits[id_gpu])

            decoder_input = decoder.build_input(
                id_gpu=id_gpu,
                tensors_input=tensors_input)
            # if in the infer, the decoder_input.input_labels and len_labels are None
            logits, preds, len_decoded = decoder(encoded, len_encoded)

            if self.training:
                if self.args.model.loss_type == 'OCD':
                    # logits = tf.Print(logits, [tensors_input.len_label_splits[id_gpu][0]], message='label length: ', summarize=1000)
                    # logits = tf.Print(logits, [tf.shape(logits[0])], message='logits shape: ', summarize=1000)
                    loss, (optimal_targets, optimal_distributions) = self.ocd_loss(
                        logits=logits,
                        len_logits=len_decoded,
                        labels=tensors_input.label_splits[id_gpu],
                        preds=preds)
                elif self.args.model.loss_type == 'CE':
                    loss = self.ce_loss(
                        logits=logits,
                        labels=decoder_input.output_labels[:, :tf.shape(logits)[1]],
                        len_labels=decoder_input.len_labels)
                elif self.args.model.loss_type == 'Premium_CE':
                    table_targets_distributions = tf.nn.softmax(tf.constant(self.args.table_targets))
                    loss = self.premium_ce_loss(
                        logits=logits,
                        labels=tensors_input.label_splits[id_gpu],
                        table_targets_distributions=table_targets_distributions,
                        len_labels=tensors_input.len_label_splits[id_gpu])
                else:
                    raise NotImplemented('NOT found loss type!')

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

        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits, preds = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)

        if preds.get_shape().ndims == 3:
            preds = preds[:,:,0]

        return preds, tensors_input.shape_batch, tf.no_op()

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
                confidence=self.args.model.decoder.label_smoothing_confidence)

            mask = tf.sequence_mask(
                len_labels,
                maxlen=tf.shape(logits)[1],
                dtype=logits.dtype)
            loss = tf.reduce_sum(crossent * mask, -1)/tf.reduce_sum(mask, -1)

        return loss

    def ocd_loss(self, logits, len_logits, labels, preds):
        from tfModels.OptimalDistill import OCD

        optimal_distributions, optimal_targets = OCD(
            hyp=preds,
            ref=labels,
            vocab_size=self.args.dim_output)

        crossent = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=optimal_distributions,
            logits=logits)

        pad_mask = tf.sequence_mask(
            len_logits,
            maxlen=tf.shape(logits)[1],
            dtype=logits.dtype)

        loss = tf.reduce_sum(crossent * pad_mask, -1)/tf.reduce_sum(pad_mask, -1)

        return loss, [optimal_targets, optimal_distributions]

    def build_idx_input(self):
        """
        used for token-input tasks such as nmt when the `self.embed_table_encoder` is given
        for the token inputs are easy to fentch form disk, there is no need to
        use tfdata.
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):

                batch_src = tf.placeholder(tf.int32, [None, None], name='input_src')
                batch_ref = tf.placeholder(tf.int32, [None, None], name='input_ref')
                batch_src_lens = tf.placeholder(tf.int32, [None], name='input_src_lens')
                batch_ref_lens = tf.placeholder(tf.int32, [None], name='input_ref_lens')
                self.list_pl = [batch_src, batch_ref, batch_src_lens, batch_ref_lens]
                # split input data alone batch axis to gpus
                embed_table = self.embed_table_encoder if self.embed_table_encoder else self.embed_table_decoder
                batch_features = tf.nn.embedding_lookup(embed_table, batch_src)
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_ref, self.num_gpus, name="label_splits")
                tensors_input.len_feat_splits = tf.split(batch_src_lens, self.num_gpus, name="len_feat_splits")
                tensors_input.len_label_splits = tf.split(batch_ref_lens, self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

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

    def build_infer_idx_input(self):
        """
        used for inference. For inference must use placeholder.
        during the infer, we only get the decoded result and not use label
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_feat_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_src = tf.placeholder(tf.int32, [None, None], name='input_src')
                batch_src_lens = tf.placeholder(tf.int32, [None], name='input_src_lens')
                self.list_pl = [batch_src, batch_src_lens]
                # split input data alone batch axis to gpus
                embed_table = self.embed_table_encoder if self.embed_table_encoder else self.embed_table_decoder
                batch_features = tf.nn.embedding_lookup(embed_table, batch_src)
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.len_feat_splits = tf.split(batch_src_lens, self.num_gpus, name="len_feat_splits")

        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def get_embedding(self, embed_table, size_input, size_embedding):
        '''
        if embed_table is not none, return it; else we will create a trainable one
        '''
        if size_embedding and (type(embed_table) is not tf.Variable):
            assert size_input
            with tf.device("/cpu:0"):
                # with tf.variable_scope(self.name, reuse=(self.__class__.num_Model > 0)):
                with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
                    embed_table = tf.get_variable(
                        "embedding", [size_input, size_embedding], dtype=tf.float32)

        return embed_table
