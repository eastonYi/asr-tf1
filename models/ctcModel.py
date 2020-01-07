import tensorflow as tf
import logging
import sys

from .utils.regularization import confidence_penalty
from .utils.tools import dense_sequence_to_sparse, choose_device
from .seq2seqModel import Seq2SeqModel

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCModel(Seq2SeqModel):
    '''
    CTC model is viewed as seq2seq model with the final FC layer as decoder.
    '''
    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 batch=None, name='CTC_Model'):
        super().__init__(tensor_global_step, encoder, decoder, training, args, batch, name)

    def __call__(self, feature, len_features, shrink=False, reuse=False):
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
                logits, align, len_logits = decoder(encoded, len_encoded, None, shrink)

        return logits, align, len_logits

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, shrink=False, reuse=tf.AUTO_REUSE):
        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]
        labels = tensors_input.label_splits[id_gpu] if tensors_input.label_splits else None
        len_labels = tensors_input.len_label_splits[id_gpu] if tensors_input.len_label_splits else None

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
                1.0, mode="fan_avg", distribution="uniform"))
            logits, align, len_logits = self(
                feature,
                len_features,
                shrink=shrink,
                reuse=reuse)

            if self.training:
                loss = self.ctc_loss(
                    logits=logits,
                    len_logits=len_logits,
                    labels=labels,
                    len_labels=len_labels)

                if self.args.model.confidence_penalty:
                    cp_loss = self.args.model.confidence_penalty * confidence_penalty(logits, len_logits)
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
            return loss, gradients, [align, labels]
        else:
            return logits, align, len_logits

    def ctc_loss(self, logits, len_logits, labels, len_labels):
        """
        No valid path found: It is possible that no valid path is found if the
        activations for the targets are zero.
        """
        labels_sparse = dense_sequence_to_sparse(
            labels,
            len_labels)
        ctc_loss = tf.nn.ctc_loss(
            labels_sparse,
            logits,
            sequence_length=len_logits,
            ctc_merge_repeated=self.args.model.merge_repeat,
            ignore_longer_outputs_than_inputs=True,
            time_major=False)

        return ctc_loss

    def build_infer_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_infer_input()
        logits, decoded, len_logits = self.build_single_graph(
            id_gpu=0,
            name_gpu=self.list_gpu_devices[0],
            tensors_input=tensors_input,
            shrink=True)

        # decoded_sparse = self.ctc_decode(logits, len_logits)
        # decoded = tf.sparse_to_dense(
        #     sparse_indices=decoded_sparse.indices,
        #     output_shape=decoded_sparse.dense_shape,
        #     sparse_values=decoded_sparse.values,
        #     default_value=0,
        #     validate_indices=True)
        distribution = tf.nn.softmax(logits)

        return decoded, tensors_input.shape_batch, distribution

    def ctc_decode(self, logits, len_logits):
        beam_size = self.args.beam_size
        logits_timeMajor = tf.transpose(logits, [1, 0, 2])

        if beam_size == 1:
            decoded_sparse = tf.to_int32(tf.nn.ctc_greedy_decoder(
                logits_timeMajor,
                len_logits,
                merge_repeated=True)[0][0])
        else:
            decoded_sparse = tf.to_int32(tf.nn.ctc_beam_search_decoder(
                logits_timeMajor,
                len_logits,
                beam_width=beam_size,
                merge_repeated=True)[0][0])

        return decoded_sparse
