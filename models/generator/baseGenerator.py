import tensorflow as tf
import logging
import sys
from collections import namedtuple

from ..utils.gradientTools import average_gradients, handle_gradients
from ..utils.tools import choose_device, smoothing_cross_entropy, warmup_exponential_decay
from ..lstmModel import LSTM_Model

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class Generator():
    num_Instances = 0
    num_Model = 0
    def __init__(self, global_step, hidden, num_blocks, training, args, name='Sequence_Dependent_Generator'):
        self.global_step = global_step
        self.batch_size = int(args.text_batch_size/args.num_gpus)
        self.dim_input = args.model.dim_input
        self.max_input_len = args.max_label_len
        self.dim_hidden = hidden
        self.num_blocks = num_blocks
        self.args = args
        self.training = training
        self.name = name
        self.num_gpus = args.num_gpus
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.run_list = self.build_graph() if training else self.build_infer_graph()
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def __call__(self, seq_input, sen_len, reuse=False):
        """
        seq_input: [b, seq_len, dim_input]
        conv generator
        """
        with tf.variable_scope(self.name, reuse=reuse):
            # inputs = tf.random_normal([int(self.batch_size), self.max_input_len])
            seq_input *= tf.sequence_mask(sen_len, maxlen=self.max_input_len, dtype=tf.float32)[:, :, None]
            x = tf.layers.dense(seq_input, self.dim_hidden, use_bias=False)
            # x = tf.reshape(x, [self.batch_size, self.max_input_len, self.dim_hidden])
            for i in range(5):
                x = tf.layers.dense(x, units=self.dim_hidden, use_bias=True)
                x = tf.nn.relu(x)

            logits = tf.layers.dense(x, units=self.args.dim_output, use_bias=False)

        return logits, sen_len

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):
        features = tensors_input.seq_input[id_gpu]
        len_features = tensors_input.seq_len[id_gpu]
        labels = tensors_input.labels[id_gpu]
        len_labels = tensors_input.len_labels[id_gpu]

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
        # with tf.device(self.center_device):
            logits, len_logits = self(features, len_features, reuse=reuse)
            loss = self.ce_loss(logits, labels, len_labels)
            loss = tf.reduce_mean(loss, -1)
            # loss = tf.reduce_mean(logits)

        if self.training:
            with tf.name_scope("gradients"):
                # var_list=self.trainable_variables()
                gradients = self.optimizer.compute_gradients(loss)
            # import pdb; pdb.set_trace()

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        if self.training:
            return loss, gradients
        else:
            return logits, len_logits

    def build_graph(self):
        tensors_input = self.build_input()
        self.build_optimizer()
        loss_step = []
        tower_grads = []

        for id_gpu, name_gpu in enumerate(self.list_gpu_devices):
            loss, gradients = self.build_single_graph(id_gpu, name_gpu, tensors_input)
            loss_step.append(loss)
            tower_grads.append(gradients)
        # mean the loss
        loss = tf.reduce_mean(loss_step)
        # merge gradients, update current model
        with tf.device(self.center_device):
            # computation relevant to gradient
            averaged_grads = average_gradients(tower_grads)
            handled_grads = handle_gradients(averaged_grads, self.args)
            op_optimize = self.optimizer.apply_gradients(handled_grads, self.global_step)

        return loss, op_optimize

    def build_infer_graph(self):
        tensors_input = self.build_input()

        logits, len_logits = self.build_single_graph(0, self.list_gpu_devices[0], tensors_input)
        decoded = tf.argmax(logits, -1)

        return logits, decoded, len_logits

    def build_input(self):
        tensors_input = namedtuple('tensors_input',
                                   'seq_input, seq_len')

        seq_input = tf.placeholder(tf.float32, [None, self.max_input_len, self.dim_input],
                                   name='seq_input')
        seq_len = tf.placeholder(tf.int32, [None],
                                 name='seq_input')
        labels = tf.placeholder(tf.int32, [None, self.max_input_len],
                                   name='labels')
        len_labels = tf.placeholder(tf.int32, [None],
                                 name='len_labels')
        self.list_pl = [seq_input, seq_len, labels, len_labels]

        tensors_input.seq_input = tf.split(seq_input, self.num_gpus, name="seq_input_splits")
        tensors_input.seq_len = tf.split(seq_len, self.num_gpus, name="seq_len_splits")
        tensors_input.labels = tf.split(labels, self.num_gpus, name="label_splits")
        tensors_input.len_labels = tf.split(seq_len, self.num_gpus, name="len_label_splits")

        return tensors_input


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
            #     confidence=1.0)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=logits)
            mask = tf.sequence_mask(
                len_labels,
                maxlen=tf.shape(logits)[1],
                dtype=loss.dtype)
            loss = tf.reduce_sum(loss * mask, -1)/tf.reduce_sum(mask, -1)

        return loss

    def trainable_variables(self, scope=None):
        '''get a list of the models's variables'''
        scope = scope if scope else self.name
        scope += '/'
        print('all the variables in the scope:', scope)
        variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES,
            scope=scope)

        return variables

    def build_optimizer(self):
        # if self.args.lr_type == 'constant_learning_rate':
        self.learning_rate = tf.convert_to_tensor(self.args.lr_G)
        # self.learning_rate = warmup_exponential_decay(
        #     self.global_step,
        #     warmup_steps=self.args.warmup_steps,
        #     peak=self.args.peak,
        #     decay_rate=0.5,
        #     decay_steps=self.args.decay_steps)

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate,
                                               beta1=0.5,
                                               beta2=0.9,
                                               epsilon=1e-9,
                                               name=self.args.optimizer)

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
