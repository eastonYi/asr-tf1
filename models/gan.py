import tensorflow as tf
import logging
from collections import namedtuple

from .utils.gradientTools import average_gradients, handle_gradients

class GAN:

    def __init__(self, tensor_global_step, G, D, name, args):
        self.G = G
        self.D = D
        self.name = name
        self.args = args
        self.center_device = "/cpu:0"
        self.tensors_input = self.build_input()

    def build_graph(self):
        tensors_input = self.build_input()
        self.optimizer_D = self.build_optimizer()
        self.optimizer_G = self.build_optimizer()

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
        with tf.device():
            # computation relevant to gradient
            averaged_D_grads = average_gradients(tower_D_grads)
            handled_D_grads = handle_gradients(averaged_D_grads, self.args)
            op_optimize_D = self.optimizer_D.apply_gradients(handled_D_grads, self.global_step)
            averaged_G_grads = average_gradients(tower_G_grads)
            handled_G_grads = handle_gradients(averaged_G_grads, self.args)
            op_optimize_G = self.optimizer_G.apply_gradients(handled_G_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(
            self.__class__.num_Instances, self.__class__.__name__))

        return loss_D, loss_G, tensors_input.shape_batch, op_optimize_D, op_optimize_G

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        with tf.device(name_gpu), tf.name_scope('GAN_split_'+str(id_gpu)):
            logits_G, loss_G_asr = self.G.build_single_graph(id_gpu, name_gpu, tensors_input.feature_splits[id_gpu])
            loss_D_asr = self.D.build_single_graph(id_gpu, name_gpu, logits_G)
            loss_D_text = self.D.build_single_graph(id_gpu, name_gpu, tensors_input.text_splits[id_gpu])
            loss_D = loss_D_text + loss_D_asr
            loss_G = loss_G_asr - loss_D_asr

            with tf.name_scope("gradients"):
                assert loss_D.get_shape().ndims == 1, loss_G.get_shape().ndims == 1
                loss_D = tf.reduce_mean(loss_D)
                loss_G = tf.reduce_mean(loss_G)
                gradients_D = self.optimizer.compute_gradients(loss_D)
                gradients_G = self.optimizer.compute_gradients(loss_G)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return loss_D, loss_G, gradients_D, gradients_G


    def build_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, \
            len_feat_splits, len_label_splits, \
            text_splits, len_text_splits, shape_batch')

        with tf.device(self.center_device), tf.name_scope("inputs"):
            # split input data alone batch axis to gpus
            tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
            tensors_input.label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
            tensors_input.text_splits = tf.split(self.batch[2], self.num_gpus, name="text_splits")
            tensors_input.len_feat_splits = tf.split(self.batch[3], self.num_gpus, name="len_feat_splits")
            tensors_input.len_label_splits = tf.split(self.batch[4], self.num_gpus, name="len_label_splits")
            tensors_input.len_text_splits = tf.split(self.batch[5], self.num_gpus, name="len_text_splits")
        tensors_input.shape_batch = tf.shape(self.batch[0])

        return tensors_input
