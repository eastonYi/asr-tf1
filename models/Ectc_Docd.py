import tensorflow as tf
import logging
import sys

from .utils.regularization import confidence_penalty
from .utils.blocks import shrink_layer
from .utils.tools import dense_sequence_to_sparse, choose_device, smoothing_cross_entropy
from .ctcModel import CTCModel
from .decoders.fc_decoder import FCDecoder

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class Ectc_Docd(CTCModel):
    '''
    CTC model is viewed as seq2seq model with the final FC layer as decoder.
    '''
    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 batch=None, name='Ectc_Docd'):
        super().__init__(tensor_global_step, encoder, decoder, training, args, batch, name)

    def __call__(self, feature, len_features, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            encoder = self.gen_encoder(
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
                    encoded, len_encoded, None, shrink=False, dim_output=self.args.dim_output)

            encoded_shrunk, len_encoded_shrunk = shrink_layer(
                encoded, len_encoded, logits_ctc, self.args.model.encoder.hidden_size)
            encoded_shrunk = tf.stop_gradient(encoded_shrunk)
            len_encoded_shrunk = tf.stop_gradient(len_encoded_shrunk)

            with tf.variable_scope(decoder.name or 'decoder'):
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

                ce_loss = self.ce_loss(
                    logits=logits_ocd[:, :min_len, :],
                    labels=labels[:, :min_len],
                    len_labels=len_labels)

                # loss = ctc_loss + ce_loss
                loss = ce_loss
                # loss = ctc_loss

                if self.args.model.confidence_penalty:
                    cp_loss = self.args.model.decoder.confidence_penalty * confidence_penalty(logits_ocd, len_logits_ocd)
                    assert cp_loss.get_shape().ndims == 1
                    loss += cp_loss

                with tf.name_scope("gradients"):
                    assert loss.get_shape().ndims == 1
                    loss = tf.reduce_mean(loss)
                    gradients = self.optimizer.compute_gradients(loss)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.training:
            return loss, gradients, [ctc_loss, ce_loss, align, labels, x]
        else:
            # return logits_ctc, decoded, len_logits_ctc
            return (logits_ctc, logits_ocd), (align, decoded), (len_logits_ctc, len_logits_ocd)

    def ce_loss(self, logits, labels, len_labels):
        """
        Compute optimization loss.
        batch major
        """
        with tf.name_scope('CE_loss'):
            # crossent = smoothing_cross_entropy(
            #     logits=logits,
            #     labels=labels,
            #     vocab_size=self.args.dim_output,
            #     confidence=self.args.model.decoder.label_smoothing)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
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
