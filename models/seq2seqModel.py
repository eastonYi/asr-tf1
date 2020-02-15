'''@file model.py
'''

import tensorflow as tf
import logging
from collections import namedtuple

from .utils.gradientTools import average_gradients, handle_gradients
from .utils.tools import right_shift_rows, choose_device, smoothing_cross_entropy, \
    get_tensor_len, warmup_exponential_decay

SOS_IDX = 2

class Seq2SeqModel(object):
    '''a general class for an encoder decoder system
    '''
    num_Instances = 0
    num_Model = 0
    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 batch=None, name='seq2seqModel'):
        '''Model constructo
        Args:
        '''
        self.name = name
        self.args = args
        self.training = training
        self.gen_encoder = encoder # encoder class
        self.gen_decoder = decoder # decoder class
        self.num_gpus = args.num_gpus if training else 1
        self.list_gpu_devices = args.list_gpus
        self.learning_rate = None
        self.batch = batch
        self.build_input = self.build_tf_input if batch else self.build_pl_input
        self.list_pl = None
        self.global_step = tensor_global_step

        # Build graph
        self.list_run = list(self.build_graph() if training else self.build_infer_graph())

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
                labels_sos = right_shift_rows(
                    p=labels,
                    shift=1,
                    pad=SOS_IDX)
                logits, preds, len_decoded = decoder(encoded, len_encoded, labels_sos)

        return logits, preds, len_decoded

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):
        """
        It worth noting that tensors_input.len_label_splits need to add 1 along with the sos padding
        """
        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]
        labels = tensors_input.label_splits[id_gpu] if tensors_input.label_splits else None
        len_labels = get_tensor_len(labels) if tensors_input.len_label_splits else None

        with tf.device(lambda op: choose_device(op, name_gpu, "/cpu:0")):

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

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()

        loss_step = []
        tower_grads = []
        list_debug = []

        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            loss, gradients, debug = self.build_single_graph(
                id_gpu, name_gpu, tensors_input)
            loss_step.append(loss)
            tower_grads.append(gradients)
            list_debug.append(debug)

        # mean the loss
        loss = tf.reduce_mean(loss_step)
        # merge gradients, update current model
        with tf.device("/cpu:0"):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(
            self.__class__.num_Instances, self.__class__.__name__))

        # return loss, tensors_input.shape_batch, op_optimize
        return loss, tensors_input.shape_batch, op_optimize, [x for x in zip(*list_debug)]

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
            loss = tf.reduce_sum(crossent * mask, -1) / tf.reduce_sum(mask, -1)

        return loss

    def build_infer_input(self):
        """
        used for inference. For inference must use placeholder.
        during the infer, we only get the decoded result and not use label
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device("/cpu:0"):
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

    def build_pl_input(self):
        """
        use for training. but recomend to use build_tf_input insted
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device("/cpu:0"):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.data.dim_input], name='input_feature')
                batch_labels = tf.placeholder(tf.int32, [None, None], name='input_labels')
                batch_feat_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                batch_label_lens = tf.placeholder(tf.int32, [None], name='input_label_lens')
                self.list_pl = [batch_features, batch_labels, batch_feat_lens, batch_label_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(batch_labels, self.num_gpus, name="label_splits")
                tensors_input.len_feat_splits = tf.split(batch_feat_lens, self.num_gpus, name="len_feat_splits")
                tensors_input.len_label_splits = tf.split(batch_label_lens, self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def build_tf_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device("/cpu:0"):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
                tensors_input.len_feat_splits = tf.split(self.batch[2], self.num_gpus, name="len_feat_splits")
                tensors_input.len_label_splits = tf.split(self.batch[3], self.num_gpus, name="len_label_splits")
        tensors_input.shape_batch = tf.shape(self.batch[0])

        return tensors_input

    def build_optimizer(self):
        if self.args.lr_type == 'constant_learning_rate':
            self.learning_rate = tf.convert_to_tensor(self.args.lr)
        else:
            self.learning_rate = warmup_exponential_decay(
                self.global_step,
                warmup_steps=self.args.warmup_steps,
                peak=self.args.peak,
                decay_rate=0.5,
                decay_steps=self.args.decay_steps)

        with tf.name_scope("optimizer"):
            if self.args.optimizer == "adam":
                logging.info("Using ADAM as optimizer")
                optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                                   beta1=0.9,
                                                   beta2=0.98,
                                                   epsilon=1e-9,
                                                   name=self.args.optimizer)
            elif self.args.optimizer == "adagrad":
                logging.info("Using Adagrad as optimizer")
                optimizer = tf.train.AdagradOptimizer(self.learning_rate)
            else:
                logging.info("Using SGD as optimizer")
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate,
                                                              name=self.args.optimizer)
        return optimizer

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

    def trainable_variables(self, scope=None):
        '''
            get a list of the models's variables
            self.trainable_variables =
            tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)
        '''
        scope = scope if scope else self.name
        scope += '/'
        logging.info('all the variables in the scope: {}'.format(scope))
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)

        return variables
