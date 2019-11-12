import tensorflow as tf
import logging
import sys
from collections import namedtuple

from .utils.gradientTools import average_gradients, handle_gradients
from .utils.tools import warmup_exponential_decay, choose_device, exponential_decay
from .encoders.blstm import BLSTM


class LSTM_Model(object):
    num_Instances = 0
    num_Model = 0

    def __init__(self, tensor_global_step, training, args, batch=None, name='model'):
        # Initialize some parameters
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

        # Build graph
        self.list_run = list(self.build_graph() if training else self.build_infer_graph())
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def build_graph(self):
        # cerate input tensors in the cpu
        tensors_input = self.build_input()
        # create optimizer
        self.optimizer = self.build_optimizer()
        if 'horovod' in sys.modules:
            import horovod.tensorflow as hvd
            logging.info('wrap the optimizer with horovod!')
            self.optimizer = hvd.DistributedOptimizer(self.optimizer)

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
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        self.__class__.num_Instances += 1
        logging.info("built {} {} instance(s).".format(
            self.__class__.num_Instances, self.__class__.__name__))

        # return loss, tensors_input.shape_batch, op_optimize
        return loss, tensors_input.shape_batch, op_optimize, [x for x in zip(*list_debug)]
        # return loss, tensors_input.shape_batch, op_optimize, debug

    def build_infer_graph(self):
        """
        reuse=True if build train models above
        reuse=False if in the inder file
        """
        # cerate input tensors in the cpu
        tensors_input = self.build_input()

        loss, logits = self.build_single_graph(
            id_gpu=0,
            name_gpu=self.list_gpu_devices[0],
            tensors_input=tensors_input,
            reuse=tf.AUTO_REUSE)

        # TODO: havn't checked
        infer = tf.nn.in_top_k(logits, tf.reshape(tensors_input.label_splits[0], [-1]), 1)

        return loss, tensors_input.shape_batch, infer

    def build_pl_input(self):
        """
        use for training. but recomend to use build_tf_input insted
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
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

    def build_infer_input(self):
        """
        used for inference. For inference must use placeholder.
        during the infer, we only get the decoded result and not use label
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, len_feat_splits, label_splits, len_label_splits, shape_batch')

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
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

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
        tensors_input.shape_batch = tf.shape(self.batch[0])

        return tensors_input

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        """
        be used for build infer model and the train model, conditioned on self.training
        """
        # build model in one device
        hidden_output = tensors_input.feature_splits[id_gpu]
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            blstm = BLSTM(args=self.args, training=self.training, name=name_gpu)
            hidden_output = blstm(hidden_output)
            logits = tf.layers.dense(inputs=hidden_output,
                                     units=self.args.dim_output,
                                     activation=tf.identity,
                                     name='fully_connected')

            # Accuracy
            with tf.name_scope("label_accuracy"):
                correct = tf.nn.in_top_k(logits, tf.reshape(tensors_input.label_splits[id_gpu], [-1]), 1)
                correct = tf.multiply(tf.cast(correct, tf.float32), tf.reshape(tensors_input.mask_splits[id_gpu], [-1]))
                label_accuracy = tf.reduce_sum(correct)
            # Cross entropy loss
            with tf.name_scope("CE_loss"):
                cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tf.reshape(tensors_input.label_splits[id_gpu], [-1]),
                    logits=logits)
                cross_entropy = tf.multiply(cross_entropy, tf.reshape(tensors_input.mask_splits[id_gpu], [-1]))
                cross_entropy_loss = tf.reduce_sum(cross_entropy) / tf.reduce_sum(tensors_input.mask_splits[id_gpu])
                loss = cross_entropy_loss

            if self.training:
                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(
                        loss, var_list=self.trainable_variables)

        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Instances))

        return loss, gradients if self.training else logits

    def build_optimizer(self):
        if self.args.lr_type == 'constant_learning_rate':
            self.learning_rate = tf.convert_to_tensor(self.args.lr)
        elif self.args.lr_type == 'exponential_decay':
            self.learning_rate = exponential_decay(
                self.global_step,
                lr_init=self.args.lr_init,
                lr_final=self.args.lr_final,
                decay_rate=self.args.decay_rate,
                decay_steps=self.args.decay_steps)
        else:
            self.learning_rate = warmup_exponential_decay(
                self.global_step,
                warmup_steps=self.args.warmup_steps,
                peak=self.args.peak,
                decay_rate=0.5,
                decay_steps=self.args.decay_steps)

        if 'horovod' in sys.modules:
            import horovod.tensorflow as hvd
            logging.info('wrap the optimizer with horovod!')
            self.learning_rate = self.learning_rate * hvd.size()

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

    def variables(self, scope=None):
        '''get a list of the models's variables'''
        scope = scope if scope else self.name
        scope += '/'
        print('all the variables in the scope:', scope)
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=scope)

        return variables

    def trainable_variables(self, scope=None):
        '''get a list of the models's variables'''
        scope = scope if scope else self.name
        scope += '/'
        print('all the variables in the scope:', scope)
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)

        return variables

if __name__ == '__main__':
    LSTM_Model()
