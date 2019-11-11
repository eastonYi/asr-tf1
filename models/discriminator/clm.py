import tensorflow as tf
from collections import namedtuple

from ..lstmModel import LSTM_Model
from ..utils.tools import choose_device


class CLM(LSTM_Model):
    """
    place_holder as input tensor: (b, t, v)
    if ndim of input is 2, then convert it to onehot
    """
    def __init__(self, tensor_global_step, training, name, args):
        self.dim_hidden = args.model_D.num_hidden
        self.max_input_len = args.max_label_len
        self.num_blocks = args.model_D.num_blocks
        self.training = training
        self.name = name
        self.args = args
        super().__init__(tensor_global_step, training, args, batch=None, name=name)

    def __call__(self, inputs, len_inputs):
        len_x = self.max_input_len
        inputs *= tf.sequence_mask(len_inputs, maxlen=len_x, dtype=tf.float32)[:, :, None]
        x = tf.layers.dense(inputs, units=self.dim_hidden, use_bias=False)
        for i in range(self.num_blocks):
            inputs = x
            x = tf.layers.conv1d(x, filters=self.dim_hidden, kernel_size=3, strides=1, padding='same')
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(x, filters=self.dim_hidden, kernel_size=3, strides=1, padding='same')
            x = tf.nn.relu(x)

            x = inputs + 1.0*x
            x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='same')
            len_x = tf.cast(tf.math.ceil(tf.cast(len_x, tf.float32)/2), tf.int32)

        x = tf.reshape(x, [-1, len_x*self.dim_hidden])
        logits = tf.layers.dense(x, units=1, use_bias=False)

        return logits

    def build_single_graph(self, id_gpu, name_gpu, tensors_input):
        with tf.device(lambda op: choose_device(op, name_gpu, self.center_device)):
            inputs = tensors_input.feature_splits[id_gpu]
            len_inputs = tensors_input.len_feat_splits[id_gpu]
            logits = self(inputs, len_inputs)
            loss = tf.reduce_mean(logits)

            with tf.name_scope("gradients"):
                gradients = self.optimizer.compute_gradients(loss, var_list=self.trainable_variables())

        return loss, gradients, [tf.no_op()]

    def build_pl_input(self):
        """
        sequence classification, so there is no label len
        """
        tensors_input = namedtuple('tensors_input',
            'feature_splits, label_splits, len_feat_splits, len_label_splits, shape_batch')

        with tf.device(self.center_device):
            with tf.name_scope("inputs"):
                batch_features = tf.placeholder(tf.float32, [None, None, self.args.dim_output], name='input_feature')
                batch_feat_lens = tf.placeholder(tf.int32, [None], name='input_fea_lens')
                self.list_pl = [batch_features, batch_feat_lens]
                # split input data alone batch axis to gpus
                tensors_input.feature_splits = tf.split(batch_features, self.num_gpus, name="feature_splits")
                tensors_input.len_feat_splits = tf.split(batch_feat_lens, self.num_gpus, name="len_feat_splits")
        tensors_input.shape_batch = tf.shape(batch_features)

        return tensors_input

    def gradient_penalty(self, real, fake, len_inputs):

        batch_size = tf.shape(real)[0]
        epsilon = tf.random_uniform([batch_size, 1, 1], minval=0., maxval=1.)
        interpolated = (1-epsilon) * real + epsilon * (fake - real)
        pred = self(interpolated, len_inputs)
        grad = tf.gradients(pred, interpolated)
        # norm = tf.norm(tf.reshape(grad, [tf.shape(grad)[0], -1]), axis=1)
        norm = tf.sqrt(1e-8 + tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
        gp = tf.reduce_mean((norm - 1.)**2)

        return gp
