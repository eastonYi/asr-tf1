import tensorflow as tf
import logging
from collections import namedtuple

from .utils.gradientTools import average_gradients, handle_gradients


class GAN:
    num_Instances = 0
    num_Model = 0
    def __init__(self, tensor_global_step, G, D, batch, name, args):
        """
        G and D are objs, they need to be created before GAN obj
        feature is a list: [feat, len_feat]
        """
        self.G = G
        self.D = D
        self.batch = batch
        self.global_step = tensor_global_step
        self.num_gpus = args.num_gpus
        self.list_gpu_devices = args.list_gpus
        self.name = name
        self.args = args
        self.center_device = "/cpu:0"
        self.list_train_G, self.list_train_D, self.list_info = self.build_graph()

    def build_graph(self):
        # tf.get_default_graph().get_name_scope().reuse_variables()
        tf.get_variable_scope().reuse_variables()
        tensors_input = self.build_input()

        loss_D_step = []; loss_G_step = []
        tower_D_grads = []; tower_G_grads = []

        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            with tf.name_scope(self.name):
                loss_D, loss_G, gradients_D, gradients_G = \
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
            # with tf.variable_scope(tf.get_default_graph().get_name_scope(), reuse=True):
            op_optimize_D = self.D.optimizer.apply_gradients(handled_D_grads, self.global_step)
            averaged_G_grads = average_gradients(tower_G_grads)
            handled_G_grads = handle_gradients(averaged_G_grads, self.args)
            # with tf.variable_scope(tf.get_default_graph().get_name_scope(), reuse=True):
            op_optimize_G = self.G.optimizer.apply_gradients(handled_G_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(
            self.__class__.num_Instances, self.__class__.__name__))

        return (loss_G, op_optimize_G), \
                (loss_D, op_optimize_D), \
                (tensors_input.shape_feature, tensors_input.shape_text)

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):

        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]
        text = tensors_input.text_splits[id_gpu]
        len_text = tensors_input.len_text_splits[id_gpu]

        with tf.device(name_gpu):
            with tf.variable_scope(self.G.name, reuse=True):
                logits_G, preds, len_decoded = self.G(feature, len_features)
            with tf.variable_scope(self.D.name, reuse=True):
                logits_D_res = self.D(logits_G, len_decoded)
                loss_D_res = tf.reduce_mean(logits_D_res)
            with tf.variable_scope(self.D.name, reuse=True):
                feature_text = tf.one_hot(text, self.args.dim_output)
                logits_D_text = self.D(feature_text, len_text)
                loss_D_text = -tf.reduce_mean(logits_D_text)

            loss_D = loss_D_text + loss_D_res
            loss_G = -loss_D_res

            with tf.name_scope("gradients"):
                gradients_D = self.D.optimizer.compute_gradients(
                    loss_D, var_list=self.D.trainable_variables)
                gradients_G = self.G.optimizer.compute_gradients(
                    loss_G, var_list=self.G.trainable_variables)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return loss_D, loss_G, gradients_D, gradients_G

    def build_input(self):
        """
        stand training input
        only use feature rather label in GAN training
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_feat_splits, text_splits, len_text_splits, shape_batch')

        with tf.device(self.center_device), tf.name_scope("GAN_inputs"):
            batch_text = tf.placeholder(tf.int32, [None, None], name='input_labels')
            batch_text_lens = tf.placeholder(tf.int32, [None], name='input_label_lens')
            self.list_pl = [batch_text, batch_text_lens]
            # split input data alone batch axis to gpus
            tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
            tensors_input.len_feat_splits = tf.split(self.batch[2], self.num_gpus, name="len_feat_splits")
            tensors_input.text_splits = tf.split(batch_text, self.num_gpus, name="text_splits")
            tensors_input.len_text_splits = tf.split(batch_text_lens, self.num_gpus, name="len_text_splits")
        tensors_input.shape_feature = tf.shape(self.batch[0])
        tensors_input.shape_text = tf.shape(batch_text)

        return tensors_input
