import tensorflow as tf
import logging
import sys

from .utils.regularization import confidence_penalty
from .utils.tools import dense_sequence_to_sparse, choose_device, get_tensor_len
from .seq2seqModel import Seq2SeqModel

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class Classifier(Seq2SeqModel):
    '''
    CTC model is viewed as seq2seq model with the final FC layer as decoder.
    '''

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
            1.0, mode="fan_avg", distribution="uniform"))
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # create encoder obj
            encoder = self.gen_encoder(
                training=self.training,
                args=self.args)
            decoder = self.gen_decoder(
                training=self.training,
                embed_table=None,
                global_step=self.global_step,
                args=self.args)

            # using encoder to encode the inout sequence
            encoded, len_encoded = encoder(
                tensors_input.feature_splits[id_gpu],
                tensors_input.len_feat_splits[id_gpu])
            logits, preds, len_decoded = decoder(encoded, len_encoded)

            if self.training:
                loss = self.CE_loss(
                    logits=logits,
                    len_logits=len_decoded,
                    labels=tensors_input.label_splits[id_gpu],
                    vocab_size=self.args.dim_output)

                if self.args.model.confidence_penalty:
                    cp_loss = self.args.model.decoder.confidence_penalty * confidence_penalty(logits, len_decoded)
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
            return loss, gradients, [preds, tensors_input.label_splits[id_gpu]]
        else:
            return logits, len_decoded

    def CE_loss(self, logits, len_logits, labels, vocab_size, confidence=0.9):
        labels = tf.tile(labels[:, 0][:, None], [1, tf.shape(logits)[1]])

        low_confidence = (1.0 - confidence) / tf.cast(vocab_size-1, tf.float32)
        normalizing = -(confidence*tf.math.log(confidence) +
            tf.cast(vocab_size-1, tf.float32) * low_confidence * tf.math.log(low_confidence + 1e-20))
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32),
            depth=vocab_size,
            on_value=confidence,
            off_value=low_confidence)

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=soft_targets)
        loss = xentropy - normalizing

        gen_loss = tf.sequence_mask(len_logits, dtype=tf.float32) * loss
        loss = tf.reduce_sum(gen_loss, -1) / tf.cast(len_logits, tf.float32)

        return loss

    def build_infer_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_infer_input()
        with tf.variable_scope(self.name, reuse=bool(self.__class__.num_Model)):
            logits, len_logits = self.build_single_graph(
                id_gpu=0,
                name_gpu=self.list_gpu_devices[0],
                tensors_input=tensors_input)
            _y = tf.argmax(tf.math.bincount(tf.argmax(logits, -1, output_type=tf.int32)))

        return _y, tensors_input.shape_batch, tf.nn.softmax(logits)
