import tensorflow as tf
import logging
import sys
from collections import namedtuple

from .utils.regularization import confidence_penalty
from .utils.blocks import shrink_layer
from .utils.tfAudioTools import batch_splice
from .utils.tools import dense_sequence_to_sparse, choose_device, smoothing_cross_entropy
from .ctcModel import CTCModel
from .decoders.fc_decoder import FCDecoder

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class Ectc_Docd(CTCModel):
    '''
    CTC model is viewed as seq2seq model with the final FC layer as decoder.
    '''
    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 encoder2=None, batch=None, name='Ectc_Docd'):
        self.gen_encoder2 = encoder2 # encoder class
        super().__init__(tensor_global_step, encoder, decoder, training, args, batch, name)

    def __call__(self, feature, len_features, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            encoder = self.gen_encoder(
                training=self.training,
                args=self.args)
            if self.gen_encoder2:
                encoder2 = self.gen_encoder2(
                    training=self.training,
                    args=self.args)
            decoder0 = FCDecoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args,
                name='ctc_decoder')
            decoder = self.gen_decoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args,
                name='ocd_decoder')

            with tf.variable_scope(encoder.name or 'encoder'):
                encoded, len_encoded = encoder(feature, len_features)

            with tf.variable_scope(decoder0.name or 'decoder0'):
                logits_ctc, align, len_logits_ctc = decoder0(
                    encoded, len_encoded, None,
                    shrink=False,
                    num_fc=self.args.model.decoder.num_fc,
                    hidden_size=self.args.model.decoder.hidden_size,
                    dim_output=self.args.dim_output)

            # shrink layer
            # encoded = batch_splice(feature, 5, 5)
            with tf.variable_scope(decoder.name or 'decoder'):
                if self.args.model.encoder2:
                    encoded, len_encoded = encoder2(feature, len_features)
                encoded_shrunk, len_encoded_shrunk = shrink_layer(
                    encoded, len_encoded, logits_ctc, encoded.get_shape()[-1])
                if not self.args.model.decoder.half:
                    encoded_shrunk = batch_splice(encoded_shrunk, 0, 0, jump=True)
                    len_encoded_shrunk = tf.cast(tf.ceil(tf.cast(len_encoded_shrunk, tf.float32)/2), tf.int32)
                logits_ocd, decoded, len_logits_ocd = decoder(encoded_shrunk, len_encoded_shrunk, None)

        return [logits_ctc, logits_ocd], [align, decoded], [len_logits_ctc, len_logits_ocd]

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):
        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]
        labels = tensors_input.label_splits[id_gpu] if tensors_input.label_splits else None
        len_labels = tensors_input.len_label_splits[id_gpu] if tensors_input.len_label_splits else None

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
                1.0, mode="fan_avg", distribution="uniform"))
            [logits_ctc, logits_ocd], [align, decoded], [len_logits_ctc, len_logits_ocd] = self(
                feature,
                len_features,
                reuse=reuse)

            if self.training:
                # ctc loss
                ctc_loss = self.ctc_loss(
                    logits=logits_ctc,
                    len_logits=len_logits_ctc,
                    labels=labels,
                    len_labels=len_labels)

                # ce loss
                len_labels = tf.where(len_labels<len_logits_ocd, len_labels, len_logits_ocd)
                len_labels = tf.where(len_labels>tf.ones_like(len_labels), len_labels, tf.ones_like(len_labels))
                min_len = tf.reduce_min([tf.shape(logits_ocd)[1], tf.shape(labels)[1]])
                x = tf.reduce_mean(tf.abs(len_labels-len_logits_ocd), -1)

                # ce blk
                blk = self.args.dim_output -1
                batch_size = tf.shape(logits_ctc)[0]
                time_length = tf.shape(logits_ctc)[1]
                ce_blk_loss = self.ce_loss(
                    logits=logits_ctc,
                    labels=tf.ones([batch_size, time_length], tf.int32)*blk,
                    len_labels=tf.ones([batch_size], tf.int32)*time_length)

                ce_loss = self.ce_loss(
                    logits=logits_ocd[:, :min_len, :],
                    labels=labels[:, :min_len],
                    len_labels=len_labels)

                # loss = ctc_loss + ce_loss + 0.2*ce_blk_loss
                # loss = ctc_loss
                loss = ctc_loss + ce_loss

                if self.args.model.confidence_penalty:
                    cp_loss = self.args.model.confidence_penalty * confidence_penalty(logits_ctc, len_logits_ctc)
                    assert cp_loss.get_shape().ndims == 1
                    loss += cp_loss

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)
                        # var_list=self.trainable_variables(self.name+'/'+'ocd_decoder'))
                        # var_list=self.trainable_variables(self.name+'/'+'encoder') +
                        # self.trainable_variables(self.name+'/'+'ctc_decoder'))

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.training:
            return loss, gradients, [ctc_loss, ce_loss, align, labels, x]
        else:
            return (logits_ctc, logits_ocd), (align, decoded), (len_logits_ctc, len_logits_ocd)

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

    def build_infer_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_infer_input()

        (logits_ctc, logits_ocd), (align, decoded), (len_logits_ctc, len_logits_ocd) = self.build_single_graph(
            id_gpu=0,
            name_gpu=self.list_gpu_devices[0],
            tensors_input=tensors_input)

        decoded_sparse = self.ctc_decode(logits_ctc, len_logits_ctc)
        decoded_ctc = tf.sparse_to_dense(
            sparse_indices=decoded_sparse.indices,
            output_shape=decoded_sparse.dense_shape,
            sparse_values=decoded_sparse.values,
            default_value=0,
            validate_indices=True)
        distribution = tf.nn.softmax(logits_ocd)

        return (decoded_ctc, decoded), tensors_input.shape_batch, distribution


class Ectc_Docd_Multi(Ectc_Docd):
    '''
    multi-label Ectc_Docd
    '''
    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 encoder2=None, batch=None, name='Ectc_Docd_Multi'):
        super().__init__(tensor_global_step, encoder, decoder, training, args, encoder2, batch, name)

    def __call__(self, feature, len_features, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            encoder = self.gen_encoder(
                training=self.training,
                args=self.args)
            if self.gen_encoder2:
                encoder2 = self.gen_encoder2(
                    training=self.training,
                    args=self.args)
            decoder0 = FCDecoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args,
                name='ctc_decoder')
            decoder = self.gen_decoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args,
                name='ocd_decoder')

            with tf.variable_scope(encoder.name or 'encoder'):
                encoded, len_encoded = encoder(feature, len_features)

            with tf.variable_scope(decoder0.name or 'decoder0'):
                logits_ctc, align, len_logits_ctc = decoder0(
                    encoded, len_encoded, None,
                    shrink=False,
                    num_fc=self.args.model.decoder.num_fc,
                    hidden_size=self.args.model.decoder.hidden_size,
                    dim_output=self.args.dim_output)

            # shrink layer
            # encoded = batch_splice(feature, 5, 5)
            with tf.variable_scope(decoder.name or 'decoder'):
                if self.args.model.encoder2:
#                     encoded, len_encoded = encoder(feature, len_features)
                    encoded, len_encoded = encoder2(feature, len_features)
                encoded_shrunk, len_encoded_shrunk = shrink_layer(
                    encoded, len_encoded, logits_ctc, encoded.get_shape()[-1])
                if not self.args.model.decoder.half:
                    encoded_shrunk = batch_splice(encoded_shrunk, 0, 0, jump=True)
                    len_encoded_shrunk = tf.cast(tf.ceil(tf.cast(len_encoded_shrunk, tf.float32)/2), tf.int32)
                logits_ocd, decoded, len_logits_ocd = decoder(encoded_shrunk, len_encoded_shrunk, None)

        return [logits_ctc, logits_ocd], [align, decoded], [len_logits_ctc, len_logits_ocd]

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):
        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]
        phones = tensors_input.phone_splits[id_gpu] if tensors_input.phone_splits else None
        len_phones = tensors_input.len_phone_splits[id_gpu] if tensors_input.len_phone_splits else None
        labels = tensors_input.label_splits[id_gpu] if tensors_input.label_splits else None
        len_labels = tensors_input.len_label_splits[id_gpu] if tensors_input.len_label_splits else None

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
                1.0, mode="fan_avg", distribution="uniform"))
            [logits_ctc, logits_ocd], [align, decoded], [len_logits_ctc, len_logits_ocd] = self(
                feature,
                len_features,
                reuse=reuse)

            if self.training:
                # ctc loss
                ctc_loss = self.ctc_loss(
                    logits=logits_ctc,
                    len_logits=len_logits_ctc,
                    labels=phones,
                    len_labels=len_phones)
                if self.args.model.confidence_penalty:
                    ctc_loss += self.args.model.confidence_penalty * confidence_penalty(logits_ctc, len_logits_ctc)

                # ce loss
                len_labels = tf.where(len_labels<len_logits_ocd, len_labels, len_logits_ocd)
                len_labels = tf.where(len_labels>tf.ones_like(len_labels), len_labels, tf.ones_like(len_labels))
                min_len = tf.reduce_min([tf.shape(logits_ocd)[1], tf.shape(labels)[1]])
                x = tf.reduce_mean(tf.abs(len_labels-len_logits_ocd), -1)

                ce_loss = self.ce_loss(
                    logits=logits_ocd[:, :min_len, :],
                    labels=labels[:, :min_len],
                    len_labels=len_labels)

                loss = ctc_loss + ce_loss
                # loss = ctc_loss
                # loss = ctc_loss + ce_loss

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)
                        # var_list=self.trainable_variables(self.name+'/'+'ocd_decoder'))
                        # var_list=self.trainable_variables(self.name+'/'+'encoder') +
                        # self.trainable_variables(self.name+'/'+'ctc_decoder'))

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.training:
            return loss, gradients, [ctc_loss, ce_loss, align, labels, x]
        else:
            return (logits_ctc, logits_ocd), (align, decoded), (len_logits_ctc, len_logits_ocd)

    def build_tf_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, phone_splits, label_splits, len_feat_splits, len_phone_splits,　len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
                tensors_input.phone_splits = tf.split(self.batch[1], self.num_gpus, name="phone_splits")
                tensors_input.label_splits = tf.split(self.batch[2], self.num_gpus, name="label_splits")
                tensors_input.len_feat_splits = tf.split(self.batch[3], self.num_gpus, name="len_feat_splits")
                tensors_input.len_phone_splits = tf.split(self.batch[4], self.num_gpus, name="len_phone_splits")
                tensors_input.len_label_splits = tf.split(self.batch[5], self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(self.batch[0])

        return tensors_input

    def build_infer_input(self):
        tensors_input = namedtuple('tensors_input',
            'feature_splits, phone_splits, label_splits, len_feat_splits, len_phone_splits,　len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_feat_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                self.list_pl = [batch_features, batch_feat_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.len_feat_splits = tf.split(batch_feat_lens, self.num_gpus, name="len_feat_splits")

        tensors_input.label_splits = None
        tensors_input.len_label_splits = None
        tensors_input.phone_splits = None
        tensors_input.len_phone_splits = None
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input
