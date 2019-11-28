import tensorflow as tf
import logging
import sys
from collections import namedtuple

from ..utils.gradientTools import average_gradients, handle_gradients
from ..utils.tools import choose_device, smoothing_cross_entropy, dense_sequence_to_sparse
from ..lstmModel import LSTM_Model
from ..utils.blocks import normal_conv, block

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class Generator():
    num_Instances = 0
    num_Model = 0
    def __init__(self, global_step, hidden, num_blocks, training, args, name='Sequence_Dependent_Generator'):
        self.global_step = global_step
        self.batch_size = int(args.text_batch_size/args.num_gpus)
        self.dim_input = args.model.dim_input
        self.max_input_len = int(args.max_label_len * args.uprate)
        self.num_filters = args.model.num_filters
        self.hidden_size = hidden
        self.num_blocks = num_blocks
        self.args = args
        self.training = training
        self.name = name
        self.num_gpus = args.num_gpus
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.run_list = self.build_graph() if training else self.build_infer_graph()
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    # def __call__(self, seq_input, sen_len, reuse=False):
#
        # with tf.variable_scope(self.name, reuse=reuse):
        #     seq_input *= tf.sequence_mask(sen_len, maxlen=self.max_input_len, dtype=tf.float32)[:, :, None]
        #     x = tf.layers.dense(seq_input, self.hidden_size, use_bias=False)
        #     for i in range(2):
        #         x = tf.layers.dense(x, units=self.hidden_size, use_bias=True)
        #         x = tf.nn.relu(x)
        #     logits = tf.layers.dense(x, units=self.args.dim_output, use_bias=False)
        #
        # return logits, sen_len

    def __call__(self, seq_input, sen_len, shrink=False, reuse=False):
        """
        seq_input: [b, seq_len, dim_input]
        conv generator
        """
        with tf.variable_scope(self.name, reuse=reuse):
            seq_input *= tf.sequence_mask(sen_len, maxlen=self.max_input_len, dtype=tf.float32)[:, :, None]
            x = tf.layers.dense(seq_input, self.hidden_size, use_bias=False)
            # x = tf.reshape(x, [-1, self.max_input_len, self.hidden_size, 1])
            # for i in range(5):
            #     x = tf.layers.dense(x, units=self.hidden_size, use_bias=True)
            #     x = tf.nn.relu(x)
            # for i in range(1):
            #     x = block(x, self.num_filters, i, kernel=(7,9))
            #     # x = normal_conv(
            #     #     inputs=x,
            #     #     filter_num=self.num_filters,
            #     #     kernel=(7,9),
            #     #     stride=(1,1),
            #     #     padding='SAME',
            #     #     use_relu=True,
            #     #     name="res_"+str(i),
            #     #     norm_type=None
            #     #     )
            for i in range(2):
                inputs = x
                x = tf.layers.conv1d(x, filters=self.hidden_size, kernel_size=3, strides=1, padding='same')
                x = tf.nn.relu(x)
                x = tf.layers.conv1d(x, filters=self.hidden_size, kernel_size=3, strides=1, padding='same')
                x = tf.nn.relu(x)
                x = inputs + 0.3*x

            x = tf.reshape(x, [-1, self.max_input_len, self.hidden_size])
            for i in range(3):
                x = tf.layers.dense(x, units=self.hidden_size, use_bias=True)
                x = tf.nn.relu(x)
            logits, len_logits = self.final_layer(x, sen_len, self.args.dim_output, shrink)

        return logits, len_logits

    # def __call__(self, seq_input, sen_len, reuse=False):
    #     """
    #     seq_input: [b, seq_len, dim_input]
    #     conv generator
    #     """
    #     with tf.variable_scope(self.name, reuse=reuse):
    #         seq_input *= tf.sequence_mask(sen_len, maxlen=self.max_input_len, dtype=tf.float32)[:, :, None]
    #         x = tf.layers.dense(seq_input, self.hidden_size, use_bias=False)
    #         x = tf.reshape(x, [-1, self.max_input_len, self.hidden_size, 1])
    #         # for i in range(5):
    #         #     x = tf.layers.dense(x, units=self.hidden_size, use_bias=True)
    #         #     x = tf.nn.relu(x)
    #         for i in range(2):
    #             x = block(x, self.num_filters, i, kernel=(7,9))
    #             # x = normal_conv(
    #             #     inputs=x,
    #             #     filter_num=self.num_filters,
    #             #     kernel=(7,9),
    #             #     stride=(1,1),
    #             #     padding='SAME',
    #             #     use_relu=True,
    #             #     name="res_"+str(i),
    #             #     norm_type=None
    #             #     )
    #         x = tf.reshape(x, [-1, self.max_input_len, self.hidden_size*self.num_filters])
    #         for i in range(2):
    #             x = tf.layers.dense(x, units=self.hidden_size, use_bias=True)
    #             x = tf.nn.relu(x)
    #         logits = tf.layers.dense(x, units=self.args.dim_output, use_bias=False)
    #
    #     return logits, sen_len

    def build_single_graph(self, id_gpu, name_gpu, tensors_input, reuse=tf.AUTO_REUSE):
        features = tensors_input.seq_input[id_gpu]
        len_features = tensors_input.seq_len[id_gpu]

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            # logits, len_logits = self(features, len_features, shrink=(not self.training),  reuse=reuse)
            logits, len_logits = self(features, len_features, shrink=False,  reuse=reuse)

            if self.training:
                labels = tensors_input.labels[id_gpu]
                len_labels = tensors_input.len_labels[id_gpu]
                loss = self.ce_loss(logits, labels, len_labels)
                loss = tf.reduce_mean(loss, -1)

                with tf.name_scope("gradients"):
                    gradients = self.optimizer.compute_gradients(loss, var_list=self.trainable_variables())

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
        # decoded_sparse = self.ctc_decode(logits, len_logits)
        # decoded = tf.sparse_to_dense(
        #     sparse_indices=decoded_sparse.indices,
        #     output_shape=decoded_sparse.dense_shape,
        #     sparse_values=decoded_sparse.values,
        #     default_value=0,
        #     validate_indices=True)

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
            # ctc_merge_repeated=False,
            ctc_merge_repeated=True,
            ignore_longer_outputs_than_inputs=True,
            time_major=False)

        return ctc_loss

    def ctc_decode(self, logits, len_logits):
        logits_timeMajor = tf.transpose(logits, [1, 0, 2])

        decoded_sparse = tf.to_int32(tf.nn.ctc_greedy_decoder(
            logits_timeMajor,
            len_logits,
            merge_repeated=True)[0][0])

        return decoded_sparse

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
                                               beta1=0.01,
                                               beta2=0.1,
                                               epsilon=1e-9,
                                               name=self.args.optimizer)

        # self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)

        # self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)

    @staticmethod
    def final_layer(encoded, len_encoded, dim_output, shrink=False):
        logits = tf.layers.dense(
            inputs=encoded,
            units=dim_output,
            activation=None,
            use_bias=False,
            name='fully_connected')

        if not shrink:
            logits *= tf.tile(tf.expand_dims(tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.float32), -1),
                              [1, 1, dim_output])

            return logits, len_encoded
        else:
            # logits *= tf.tile(tf.expand_dims(tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.float32), -1),
            #                   [1, 1, dim_output])
            batch_size = tf.shape(logits)[0]
            blank_id = tf.convert_to_tensor(dim_output - 1, dtype=tf.int64)
            frames_mark = tf.not_equal(tf.argmax(logits, -1), blank_id)
            prev = tf.concat([tf.ones([batch_size, 1], tf.int64) * blank_id, tf.argmax(logits, -1)[:, :-1]], 1)
            flag_norepeat = tf.not_equal(prev, tf.argmax(logits, -1))
            flag = tf.logical_and(flag_norepeat, frames_mark)
            flag = tf.logical_and(flag, tf.sequence_mask(len_encoded, tf.shape(logits)[1], tf.bool))
            len_labels = tf.reduce_sum(tf.cast(flag, tf.int32), -1)
            max_label_len = tf.reduce_max(len_labels)
            logits_output = tf.zeros([0, max_label_len, dim_output], tf.float32)

            def sent(b, logits_output):
                logit = tf.gather(logits[b, :, :], tf.where(flag[b, :])[:, 0])
                pad_logit = tf.zeros([tf.reduce_max([max_label_len - len_labels[b], 0]), dim_output])
                logits_padded = tf.concat([logit, pad_logit], 0)[:max_label_len, :]
                logits_output = tf.concat([logits_output, logits_padded[None, :]], 0)

                return b+1, logits_output

            _, logits_output = tf.while_loop(
            cond=lambda b, *_: tf.less(b, batch_size),
            body=sent,
            loop_vars=[0, logits_output],
            shape_invariants=[tf.TensorShape([]),
                              tf.TensorShape([None, None, dim_output])])

            return logits_output, len_labels
