import tensorflow as tf
import logging
import sys

from ..utils.tools import choose_device

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class Generator():
    num_Instances = 0
    num_Model = 0
    def __init__(self, tensor_global_step, hidden, num_blocks, args, name='Gerneral_Generator_Model'):
        self.batch_size = int(args.text_batch_size/args.num_gpus)
        self.max_input_len = args.max_label_len
        self.dim_hidden = hidden
        self.num_blocks = num_blocks
        self.args = args
        self.name = name
        self.num_gpus = args.num_gpus
        self.list_gpu_devices = args.list_gpus
        self.center_device = "/cpu:0"
        self.run_list = self.build_graph()
        self.trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

    def __call__(self, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            inputs = tf.random_normal([int(self.batch_size), self.max_input_len])
            x = tf.layers.dense(inputs, units=self.max_input_len * self.dim_hidden, use_bias=False)
            x = tf.reshape(x, [self.batch_size, self.max_input_len, self.dim_hidden])
            for i in range(self.num_blocks):
                inputs = x
                x = tf.layers.conv1d(x, filters=self.dim_hidden, kernel_size=5, strides=1, padding='same')
                x = tf.nn.tanh(x)
                x = tf.layers.conv1d(x, filters=self.dim_hidden, kernel_size=5, strides=1, padding='same')
                x = tf.nn.tanh(x)
                x = inputs + 0.3*x

            logits = tf.layers.dense(x, units=self.args.dim_output, use_bias=True)

        return logits, tf.ones([self.batch_size], tf.int32) * self.max_input_len

    def build_single_graph(self, id_gpu, name_gpu, reuse=tf.AUTO_REUSE):

        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            logits, len_logits = self(reuse=reuse)

        self.__class__.num_Model += 1
        logging.info('\tbuild {} on {} succesfully! total model number: {}'.format(
            self.__class__.__name__, name_gpu, self.__class__.num_Model))

        return logits, len_logits

    def build_graph(self):
        logits, len_logits = self.build_single_graph(0, self.list_gpu_devices[0])
        decoded = tf.argmax(logits, -1)

        return decoded, len_logits
