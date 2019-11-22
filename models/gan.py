import tensorflow as tf
import logging
from collections import namedtuple

from .utils.gradientTools import average_gradients, handle_gradients
from .utils.tools import batch3D_pad_to, warmup_exponential_decay
from .utils.regularization import confidence_penalty

class GAN:
    num_Instances = 0
    num_Model = 0
    def __init__(self, tensor_global_step, G, D, batch, unbatch, name, args):
        """
        G and D are objs, they need to be created before GAN obj
        feature is a list: [feat, len_feat]
        """
        self.G = G
        self.D = D
        self.batch = batch
        self.unbatch = unbatch
        self.global_step0 = tensor_global_step[0]
        self.global_step1 = tensor_global_step[1]
        self.num_gpus = args.num_gpus
        self.list_gpu_devices = args.list_gpus
        self.name = name
        self.args = args
        self.center_device = "/cpu:0"
        self.list_train_D, self.list_train_G, self.list_feature_shape = self.build_graph()

    def build_graph(self):
        self.build_optimizer()
        tensors_input = self.build_input()

        loss_D_step = []; loss_G_step = [];
        tower_D_grads = []; tower_G_grads = []

        with tf.name_scope(self.name):
            for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
                loss_D, loss_G, gradients_D, gradients_G, (loss_G_supervise, loss_D_res, loss_D_text, loss_gp) = \
                    self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_D_step.append(loss_D); loss_G_step.append(loss_G)
                tower_D_grads.append(gradients_D); tower_G_grads.append(gradients_G)

        # mean the loss
        loss_D = tf.reduce_mean(loss_D_step); loss_G = tf.reduce_mean(loss_G_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_D_grads = average_gradients(tower_D_grads)
            handled_D_grads = handle_gradients(averaged_D_grads, self.args)
            op_optimize_D = self.optimizer_D.apply_gradients(handled_D_grads, self.global_step0)
            averaged_G_grads = average_gradients(tower_G_grads)
            handled_G_grads = handle_gradients(averaged_G_grads, self.args)
            op_optimize_G = self.optimizer_G.apply_gradients(handled_G_grads, self.global_step1)
            # op_optimize_G = self.G.optimizer.apply_gradients(handled_G_grads, self.global_step1)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(
            self.__class__.num_Instances, self.__class__.__name__))

        return (loss_D, loss_D_res, loss_D_text, loss_gp, op_optimize_D), \
                (loss_G, loss_G_supervise, op_optimize_G), \
                (tensors_input.shape_feature, tensors_input.shape_unfeature)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]
        labels = tensors_input.label_splits[id_gpu]
        len_labels = tensors_input.len_label_splits[id_gpu]

        unfeature = tensors_input.unfeature_splits[id_gpu]
        len_unfeatures = tensors_input.len_unfeat_splits[id_gpu]
        len_unlabel = tensors_input.len_unlabel_splits[id_gpu]
        text = tensors_input.text_splits[id_gpu]
        len_text = tensors_input.len_text_splits[id_gpu]

        with tf.device(name_gpu):
            # G loss
            logits_G, _, len_decoded = self.G(feature, len_features, shrink=False, reuse=True)
            loss_G_supervise = self.G.ctc_loss(
                logits=logits_G,
                len_logits=len_decoded,
                labels=labels,
                len_labels=len_labels)
            cp_loss = self.args.model.decoder.confidence_penalty * confidence_penalty(logits_G, len_decoded)
            loss_G_supervise = loss_G_supervise + cp_loss
            loss_G_supervise = tf.reduce_mean(loss_G_supervise)
            # loss_G_supervise = tf.constant(0.0)
            logits_G_un, _, len_decoded = self.G(unfeature, len_unfeatures, shrink=True, reuse=True)
            # sample_mask = tf.cast(tf.equal(len_unlabel, len_decoded), tf.float32)
            # sample_mask = tf.zeros_like(len_label, dtype=tf.float32)
            # sample_mask = tf.ones_like(len_unlabel, dtype=tf.float32)

            # D loss fake
            logits_G_un = batch3D_pad_to(logits_G_un, length=self.args.max_label_len)
            logits_D_res = self.D(tf.nn.softmax(logits_G_un, -1), len_decoded, reuse=True)
            # logits_G_un = tf.zeros([tf.shape(logits_G_un)[0], self.args.max_label_len, self.args.dim_output], tf.float32)
            # len_decoded = tf.ones([tf.shape(logits_G_un)[0]], tf.int32) * self.args.max_label_len
            # logits_D_res = self.D(logits_G_un, len_decoded, reuse=True)
            loss_D_res = tf.reduce_mean(logits_D_res, 0)

            # zeros = tf.zeros(tf.shape(logits_D_res), tf.float32)
            # ones = tf.ones(tf.shape(logits_D_res), tf.float32)
            # # loss_D_res = tf.math.pow(logits_D_res - zeros, 2)
            # loss_D_res = tf.nn.relu(logits_D_res - zeros)
            # loss_D_res = tf.reduce_sum(loss_D_res*sample_mask) / tf.reduce_sum(sample_mask)
            # loss_G_res = tf.nn.relu(ones - logits_D_res)
            # loss_G_res = tf.reduce_sum(loss_G_res*sample_mask) / tf.reduce_sum(sample_mask)

            # D loss real
            feature_text = tf.one_hot(text, self.args.dim_output)
            logits_D_text = self.D(feature_text, len_text, reuse=True)
            loss_D_text = -tf.reduce_mean(logits_D_text, 0)

            # ones = tf.ones(tf.shape(logits_D_text), tf.float32)
            # zeros = tf.zeros(tf.shape(logits_D_text), tf.float32)
            # # loss_D_text = tf.math.pow(logits_D_text - ones, 2)
            # loss_D_text = tf.nn.relu(ones - logits_D_text)
            # loss_D_text = tf.reduce_mean(loss_D_text)

            # D loss greadient penalty
            # idx = tf.random.uniform(
            #     (), maxval=(self.args.text_batch_size-self.args.batch_size), dtype=tf.int32)
            gp = 0.1 * self.D.gradient_penalty(
                # real=feature_text[idx:idx+4],
                real=feature_text[0:tf.shape(logits_G_un)[0]],
                fake=tf.nn.softmax(logits_G_un, -1),
                len_inputs=len_decoded)
            # gp = tf.constant(0.0)

            # loss_D_res = tf.constant(0.0)
            loss_D = loss_D_res + loss_D_text +  gp
            # loss_D = loss_D_res
            # loss_G = -loss_D_res
            loss_G = self.args.supervise_G_rate * loss_G_supervise - loss_D_res
            # loss_D = loss_D_res + loss_D_text
            # loss_G = loss_G_res
            # loss_G = 0

            with tf.name_scope("gradients"):
                gradients_D = self.optimizer_D.compute_gradients(
                    loss_D, var_list=self.D.trainable_variables)
                gradients_G = self.optimizer_G.compute_gradients(
                    loss_G, var_list=self.G.trainable_variables)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return loss_D, loss_G, gradients_D, gradients_G, [loss_G_supervise, loss_D_res, loss_D_text, gp]

    # def build_single_graph(self, id_gpu, name_gpu, tensors_input):
    #
    #     text = tensors_input.text_splits[id_gpu]
    #     len_text = tensors_input.len_text_splits[id_gpu]
    #
    #     with tf.device(name_gpu):
    #         # G loss
    #         batch_size = tf.shape(text)[0]
    #         logits_G_un, len_decoded = self.G(reuse=True)
    #
    #         # D loss fake
    #         logits_D_res = self.D(tf.nn.softmax(logits_G_un, -1), len_decoded, reuse=True)
    #         loss_D_res = tf.reduce_mean(logits_D_res)
    #
    #         # D loss real
    #         feature_text = tf.one_hot(text, self.args.dim_output)
    #         logits_D_text = self.D(feature_text, len_text, reuse=True)
    #         loss_D_text = -tf.reduce_mean(logits_D_text)
    #
    #         # D loss greadient penalty
    #         # idx = tf.random.uniform(
    #         #     (), maxval=(self.args.text_batch_size-self.args.batch_size), dtype=tf.int32)
    #         gp = 0.05 * self.D.gradient_penalty(
    #             # real=feature_text[idx:idx+4],
    #             real=feature_text[0:tf.shape(logits_G_un)[0]],
    #             fake=tf.nn.softmax(logits_G_un, -1),
    #             len_inputs=len_decoded)
    #         loss_G_supervise = tf.constant(0.0)
    #
    #         # loss_D_res = tf.constant(0.0)
    #         loss_D = loss_D_res + loss_D_text +  gp
    #         loss_G = -loss_D_res
    #         # loss_D = loss_D_res + loss_D_text
    #         # loss_G = loss_G_res
    #         # loss_G = 0
    #
    #         with tf.name_scope("gradients"):
    #             gradients_D = self.optimizer_D.compute_gradients(
    #                 loss_D, var_list=self.D.trainable_variables)
    #             gradients_G = self.optimizer_G.compute_gradients(
    #                 loss_G, var_list=self.G.trainable_variables)
    #
    #     self.__class__.num_Model += 1
    #     logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
    #         self.__class__.__name__, name_gpu, self.__class__.num_Model))
    #
    #     return loss_D, loss_G, gradients_D, gradients_G, [loss_G_supervise, loss_D_res, loss_D_text, gp]


    def build_input(self):
        """
        stand training input
        only use feature rather label in GAN training
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_feat_splits, text_splits, len_text_splits, shape_batch')

        with tf.device(self.center_device), tf.name_scope("GAN_inputs"):
            batch_text = tf.placeholder(tf.int32, [None, None], name='input_text')
            batch_text_lens = tf.placeholder(tf.int32, [None], name='input_len_text')
            self.list_pl = [batch_text, batch_text_lens]
            # split input data alone batch axis to gpus
            tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
            tensors_input.label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
            tensors_input.len_feat_splits = tf.split(self.batch[2], self.num_gpus, name="len_feat_splits")
            tensors_input.len_label_splits = tf.split(self.batch[3], self.num_gpus, name="len_label_splits")

            tensors_input.unfeature_splits = tf.split(self.unbatch[0][:, :self.args.model_D.max_feat_len, :], self.num_gpus, name="unfeature_splits")
            tensors_input.unlabel_splits = tf.split(self.unbatch[1], self.num_gpus, name="unlabel_splits")
            tensors_input.len_unfeat_splits = tf.split(self.unbatch[2], self.num_gpus, name="len_unfeat_splits")
            tensors_input.len_unlabel_splits = tf.split(self.unbatch[3], self.num_gpus, name="len_unlabel_splits")

            tensors_input.text_splits = tf.split(batch_text, self.num_gpus, name="text_splits")
            tensors_input.len_text_splits = tf.split(batch_text_lens, self.num_gpus, name="len_text_splits")

        tensors_input.shape_feature = tf.shape(self.batch[0])
        tensors_input.shape_unfeature = tf.shape(self.unbatch[0])

        return tensors_input

    def build_optimizer(self):
        # if self.args.lr_type == 'constant_learning_rate':
        self.learning_rate_G = tf.convert_to_tensor(self.args.lr_G)
        self.learning_rate_D = tf.convert_to_tensor(self.args.lr_D)
        # self.optimizer_G = tf.train.GradientDescentOptimizer(self.learning_rate_G)
        # self.optimizer_D = tf.train.GradientDescentOptimizer(self.learning_rate_D)
        # else:
        # self.learning_rate_G = warmup_exponential_decay(
        #     self.global_step0,
        #     warmup_steps=self.args.warmup_steps,
        #     peak=self.args.peak,
        #     decay_rate=0.5,
        #     decay_steps=self.args.decay_steps)
        #     self.learning_rate_D = warmup_exponential_decay(
        #         self.global_step1,
        #         warmup_steps=self.args.warmup_steps,
        #         peak=self.args.peak,
        #         decay_rate=0.5,
        #         decay_steps=self.args.decay_steps)

        # self.optimizer_D = tf.train.AdamOptimizer(self.learning_rate_D,
        #                                           beta1=0.5,
        #                                           beta2=0.9,
        #                                           epsilon=1e-9,
        #                                           name='optimizer_D')

        # self.optimizer_G = tf.train.AdamOptimizer(self.learning_rate_G,
        #                                           beta1=0.5,
        #                                           beta2=0.9,
        #                                           epsilon=1e-9,
        #                                           name='optimizer_G')
        self.optimizer_G = tf.train.RMSPropOptimizer(self.learning_rate_G)
        self.optimizer_D = tf.train.RMSPropOptimizer(self.learning_rate_D)


class Conditional_GAN(GAN):

    def build_graph(self):
        self.build_optimizer()
        tensors_input = self.build_input()

        loss_D_step = []; loss_G_step = [];
        tower_D_grads = []; tower_G_grads = []

        with tf.name_scope(self.name):
            for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
                loss_D, loss_G, gradients_D, gradients_G, (loss_D_res, loss_D_text, loss_gp) = \
                    self.build_single_graph(id_gpu, name_gpu, tensors_input)
                loss_D_step.append(loss_D); loss_G_step.append(loss_G)
                tower_D_grads.append(gradients_D); tower_G_grads.append(gradients_G)

        # mean the loss
        loss_D = tf.reduce_mean(loss_D_step); loss_G = tf.reduce_mean(loss_G_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_D_grads = average_gradients(tower_D_grads)
            handled_D_grads = handle_gradients(averaged_D_grads, self.args)
            op_optimize_D = self.optimizer_D.apply_gradients(handled_D_grads, self.global_step0)
            averaged_G_grads = average_gradients(tower_G_grads)
            handled_G_grads = handle_gradients(averaged_G_grads, self.args)
            op_optimize_G = self.optimizer_G.apply_gradients(handled_G_grads, self.global_step1)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(
            self.__class__.num_Instances, self.__class__.__name__))

        return (loss_D, loss_D_res, loss_D_text, loss_gp, op_optimize_D), (loss_G, op_optimize_G), tf.no_op()

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        feature = tensors_input.feature_splits[id_gpu]
        len_feature = tensors_input.len_feat_splits[id_gpu]
        text = tensors_input.text_splits[id_gpu]
        len_text = tensors_input.len_text_splits[id_gpu]

        with tf.device(name_gpu):
            # G loss
            # loss_G_supervise = tf.constant(0.0)
            logits_G_un, len_decoded = self.G(feature, len_feature, reuse=True)

            # D loss fake
            logits_G_un = batch3D_pad_to(logits_G_un, length=self.args.max_label_len)
            logits_D_res = self.D(tf.nn.softmax(logits_G_un, -1), len_decoded, reuse=True)
            loss_D_res = tf.reduce_mean(logits_D_res, 0)

            # D loss real
            feature_text = tf.one_hot(text, self.args.dim_output)
            logits_D_text = self.D(feature_text, len_text, reuse=True)
            loss_D_text = -tf.reduce_mean(logits_D_text, 0)

            gp = 10.0 * self.D.gradient_penalty(
                # real=feature_text[idx:idx+4],
                real=feature_text,
                fake=tf.nn.softmax(logits_G_un, -1),
                len_inputs=len_decoded)
            # gp = tf.constant(0.0)

            # loss_D_res = tf.constant(0.0)
            loss_D = loss_D_res + loss_D_text + gp
            # loss_D = loss_D_res
            loss_G = -loss_D_res
            # loss_D = loss_D_res + loss_D_text
            # loss_G = loss_G_res
            # loss_G = 0

            with tf.name_scope("gradients"):
                gradients_D = self.optimizer_D.compute_gradients(
                    loss_D, var_list=self.D.trainable_variables)
                gradients_G = self.optimizer_G.compute_gradients(
                    loss_G, var_list=self.G.trainable_variables)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return loss_D, loss_G, gradients_D, gradients_G, [loss_D_res, loss_D_text, gp]

    def build_input(self):
        """
        stand training input
        only use feature rather label in GAN training
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_feat_splits, text_splits, len_text_splits')

        with tf.device(self.center_device), tf.name_scope("GAN_inputs"):
            batch_feature = tf.placeholder(tf.float32, [None, None, self.args.model.dim_input], name='input_feature')
            batch_feature_len = tf.placeholder(tf.int32, [None], name='input_len_feat')
            batch_text = tf.placeholder(tf.int32, [None, None], name='input_text')
            batch_text_lens = tf.placeholder(tf.int32, [None], name='input_len_text')
            self.list_G_pl = [batch_feature, batch_feature_len]
            self.list_D_pl = [batch_text, batch_text_lens]

            tensors_input.feature_splits = tf.split(batch_feature, self.num_gpus, name="feature_splits")
            tensors_input.len_feat_splits = tf.split(batch_feature_len, self.num_gpus, name="len_feat_splits")
            tensors_input.text_splits = tf.split(batch_text, self.num_gpus, name="text_splits")
            tensors_input.len_text_splits = tf.split(batch_text_lens, self.num_gpus, name="len_text_splits")

        return tensors_input

    def build_optimizer(self):
        # if self.args.lr_type == 'constant_learning_rate':
        self.learning_rate_G = tf.convert_to_tensor(self.args.lr_G)
        self.learning_rate_D = tf.convert_to_tensor(self.args.lr_D)
        # self.optimizer_G = tf.train.GradientDescentOptimizer(self.learning_rate_G)
        # self.optimizer_D = tf.train.GradientDescentOptimizer(self.learning_rate_D)
        # else:
        # self.learning_rate_G = warmup_exponential_decay(
        #     self.global_step0,
        #     warmup_steps=self.args.warmup_steps,
        #     peak=self.args.peak,
        #     decay_rate=0.5,
        #     decay_steps=self.args.decay_steps)
        #     self.learning_rate_D = warmup_exponential_decay(
        #         self.global_step1,
        #         warmup_steps=self.args.warmup_steps,
        #         peak=self.args.peak,
        #         decay_rate=0.5,
        #         decay_steps=self.args.decay_steps)
        self.optimizer_G = tf.train.AdamOptimizer(self.learning_rate_G,
                                                   beta1=0.5,
                                                   beta2=0.9,
                                                   epsilon=1e-9)
        self.optimizer_D = tf.train.AdamOptimizer(self.learning_rate_D,
                                                   beta1=0.5,
                                                   beta2=0.9,
                                                   epsilon=1e-9)

        # self.optimizer_G = tf.train.RMSPropOptimizer(self.learning_rate_G)
        # self.optimizer_D = tf.train.RMSPropOptimizer(self.learning_rate_D)
