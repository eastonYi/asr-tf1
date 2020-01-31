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

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

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

    def schedule_sampling(self, encoder_output, Y):
        # Prepare beam search inputs.
        batch_size = tf.shape(encoder_output)[0]
        preds = tf.ones([batch_size, 1], dtype=tf.int32) * self.args.sis_idx
        # [batch_size, step=0, dst_vocab_size]
        logits = tf.zeros([batch_size, 0, self.args.dim_output])
        y_max_length = tf.shape(Y)[1]

        def not_finished(i, preds, logits):
            return tf.less(i, y_max_length)

        def step(i, preds, logits):
            last_logit = self.decoder(preds, encoder_output, reuse=True)
            # 采样概率正确值或者推理值, 第一个是teacher_force概率，第二个是inference概率, 这里只选一个值
            schedule_sampling_epsilon = tf.constant([[self.args.schedule_sampling_epsilon, 1 - self.args.schedule_sampling_epsilon]],
                dtype=tf.float32)
            # tf.multinomial()按照该概率分布进行采样, 返回值第一维是batch_size, 第二维是logits第二维上的id
            # 即返回0表示选择teacher_force，反之为inference，返回的shape为(1,1)
            sampling_result = tf.to_int32(
                tf.multinomial(tf.log(schedule_sampling_epsilon), num_samples=1, seed=None, name=None))

            def teacher_force():
                return Y[:, i]

            def sampling_inference():
                z = tf.nn.log_softmax(last_logit)
                # tf.multinomial()按照该概率分布进行采样, 返回值第一维是batch_size, 第二维是logits第二维上的id
                last_preds = tf.to_int32(tf.multinomial(z, num_samples=1, seed=None, name=None))
                return last_preds

            # 默认选择常值idx=0
            teacher_force_constant = tf.constant([[0]], dtype=tf.int32)
            # tf.equal对输入的sampler_result和one_constant逐元素做逻辑比较，返回bool类型的 Tensor,支持broadcasting
            # 这里的tf.equal(sampler_result, one_constant)结果的shape=(1,1)，例如：[[True]]
            last_preds = tf.cond(tf.equal(sampling_result, teacher_force_constant)[0][0], teacher_force,
                                 sampling_inference)

            preds = tf.concat((preds, last_preds), axis=1)  # [batch_size, step=i]
            logits = tf.concat((logits, last_logit[:, None, :]), axis=1)  # [batch_size, step=i, dst_vocab_size]

            return i+1, preds, logits

        i, preds, logits = tf.while_loop(cond=not_finished,
                                         body=step,
                                         loop_vars=[0, preds, logits],
                                         shape_invariants=[
                                             tf.TensorShape([]),
                                             tf.TensorShape([None, None]),
                                             tf.TensorShape([None, None, None])])

        preds = preds[:, 1:]  # remove <S> flag
        return preds, logits
