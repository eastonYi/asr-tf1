import tensorflow as tf
import logging
import sys
from collections import namedtuple

from .utils.gradientTools import average_gradients, handle_gradients
from .utils.tools import choose_device
from .ctcModel import CTCModel

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTCModel(CTCModel):
    '''
    CTC model is viewed as seq2seq model with the final FC layer as decoder.
    '''
    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 kernel=None, py=None, batch=None, unbatch=None, name='CTC_Model'):
        self.top_k = args.top_k
        self.ngram = args.ngram
        self.kernel = kernel
        self.py = py
        self.unbatch = unbatch
        self.name = name
        self.args = args
        self.training = training
        self.gen_encoder = encoder # encoder class
        self.gen_decoder = decoder # decoder class
        self.training = training
        self.num_gpus = args.num_gpus if training else 1
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.learning_rate = None
        self.args = args
        self.batch = batch
        self.name = name
        self.build_input = self.build_tf_input if batch else self.build_pl_input
        self.list_pl = None
        self.global_step = tensor_global_step
        if training:
            self.list_run, self.list_run_EODM = list(self.build_graph())
        else:
            self.list_run = list(self.build_infer_graph())
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

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

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):
        feature = tensors_input.feature_splits[id_gpu]
        len_features = tensors_input.len_feat_splits[id_gpu]

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            tf.get_variable_scope().set_initializer(tf.variance_scaling_initializer(
                1.0, mode="fan_avg", distribution="uniform"))
            logits, align, len_logits = self(
                feature,
                len_features,
                shrink=False,
                reuse=reuse)

            if self.training:
                # CTC loss
                labels = tensors_input.label_splits[id_gpu]
                len_labels = tensors_input.len_label_splits[id_gpu]
                loss_CTC = self.ctc_loss(
                    logits=logits,
                    len_logits=len_logits,
                    labels=labels,
                    len_labels=len_labels)
                loss_CTC = tf.reduce_mean(loss_CTC)

                # EODM loss
                unfeature = tensors_input.unfeature_splits[id_gpu]
                len_unfeatures = tensors_input.len_unfeat_splits[id_gpu]
                unlogits, unalign, len_unlogits = self(
                    unfeature,
                    len_unfeatures,
                    shrink=False,
                    reuse=reuse)
                px_batch = tf.nn.softmax(logits)
                x_log = tf.math.log(px_batch + 1e-15)
                x_conv = tf.layers.conv1d(x_log,
                                          filters=self.args.EODM.top_k,
                                          kernel_size=self.args.EODM.ngram,
                                          strides=1,
                                          padding='valid',
                                          use_bias=False,
                                          kernel_initializer=lambda *args, **kwargs: self.kernel,
                                          trainable=False,
                                          reuse=(id_gpu!=0))

                pz = tf.exp(x_conv)
                mask = tf.sequence_mask(len_logits, maxlen=tf.shape(pz)[1], dtype=tf.float32)[:, :, None]
                pz = tf.reduce_sum(pz * mask, [0, 1]) / tf.reduce_sum(mask, [0, 1]) # [z]
                loss_z = - tf.convert_to_tensor(self.py) * tf.math.log(pz+1e-15) # batch loss
                loss_EODM = tf.reduce_sum(loss_z)

                with tf.name_scope("gradients"):
                    gradients_CTC = self.optimizer.compute_gradients(loss_CTC)
                    gradients_EODM = self.optimizer.compute_gradients(loss_EODM)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.training:
            return (loss_CTC, loss_EODM), (gradients_CTC, gradients_EODM), [loss_CTC, loss_EODM]
        else:
            return logits, len_logits

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        self.optimizer = self.build_optimizer()

        loss_step = []; loss_EODM_step = []
        tower_grads = []; tower_EODM_grads = []
        list_debug = []

        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            (loss_CTC, loss_EODM), (gradients, gradients_EODM), debug = self.build_single_graph(
                id_gpu, name_gpu, tensors_input)
            loss_step.append(loss_CTC); loss_EODM_step.append(loss_EODM)
            tower_grads.append(gradients); tower_EODM_grads.append(gradients_EODM)
            list_debug.append(debug)

        # mean the loss
        loss = tf.reduce_mean(loss_step)
        loss_EODM = tf.reduce_mean(loss_EODM_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

            averaged_grads_EODM = average_gradients(tower_EODM_grads)
            handled_grads_EODM = handle_gradients(averaged_grads_EODM, self.args)
            op_optimize_EODM = self.optimizer.apply_gradients(handled_grads_EODM, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(
            self.__class__.num_Instances, self.__class__.__name__))

        # return loss, tensors_input.shape_batch, op_optimize
        return (loss, tensors_input.shape_batch, op_optimize), (loss_EODM, tensors_input.shape_unbatch, op_optimize_EODM)

    def build_tf_input(self):
        """
        stand training input
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(self.batch[0], self.num_gpus, name="feature_splits")
                tensors_input.label_splits = tf.split(self.batch[1], self.num_gpus, name="label_splits")
                tensors_input.len_feat_splits = tf.split(self.batch[2], self.num_gpus, name="len_feat_splits")
                tensors_input.len_label_splits = tf.split(self.batch[3], self.num_gpus, name="len_label_splits")
                tensors_input.unfeature_splits = tf.split(self.unbatch[0], self.num_gpus, name="unfeature_splits")
                tensors_input.len_unfeat_splits = tf.split(self.unbatch[2], self.num_gpus, name="len_unfeat_splits")
        tensors_input.shape_batch = tf.shape(self.batch[0])
        tensors_input.shape_unbatch = tf.shape(self.unbatch[0])

        return tensors_input
