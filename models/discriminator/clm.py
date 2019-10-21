import tensorflow as tf


class CLM:

    def __init__(self, tensor_global_step, training, name, args):
        self.dim_hidden = args.model.D.num_hidden
        self.num_blocks = args.model.D.num_blocks
        self.name = name
        self.args = args

    def __call__(self, inputs):
        x = tf.layers.dense(inputs, units=self.dim_hidden, use_bias=False)

        for i in range(self.num_blocks):
            inputs = x
            x = tf.layers.conv1d(x, filters=self.dim_hidden, kernel_size=3, stride=1, padding='valid')
            x = tf.nn.relu(x)
            x = tf.layers.conv1d(x, filters=self.dim_hidden, kernel_size=3, stride=1, padding='valid')
            x = tf.nn.relu(x)

            x = inputs + 1.0*x
            x = tf.layers.max_pooling1d(x, pool_size=2, strides=2, padding='same')

        _, time, hidden = x.shape
        x = tf.reshape(x, [-1, time*hidden])
        output = tf.layers.dense(inputs, units=1, use_bias=False)

        self.trainable_variables = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.name)

        return output
