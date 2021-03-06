import tensorflow as tf
import numpy as np

from .encoder import Encoder
from ..utils.blocks import conv_lstm
from ..utils.attention import layer_norm

from ..utils.attention import residual, multihead_attention, ff_hidden,\
    attention_bias_ignore_padding, add_timing_signal_1d


class Transformer_Encoder(Encoder):
    def __init__(self, args, training, name=None):
        super().__init__(args, training, name=None)
        self.attention_dropout_rate = args.model.encoder.attention_dropout_rate if training else 0.0
        self.residual_dropout_rate = args.model.encoder.residual_dropout_rate if training else 0.0
        self.hidden_units = args.model.encoder.num_cell_units
        self.num_heads = args.model.encoder.num_heads
        self.num_blocks = args.model.encoder.num_blocks
        self._ff_activation = lambda x, y: x * tf.sigmoid(y)

    def __call__(self, features, len_features):

        encoder_output = tf.layers.dense(
            inputs=features,
            units=self.hidden_units,
            activation=None,
            use_bias=False,
            name='encoder_fc')
        encoder_output = tf.contrib.layers.layer_norm(
            encoder_output, center=True, scale=True, trainable=True)

        # Add positional signal
        encoder_output = add_timing_signal_1d(encoder_output)
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=self.residual_dropout_rate,
                                           training=self.training)
        # Mask
        encoder_padding = tf.equal(tf.sequence_mask(len_features, maxlen=tf.shape(features)[1]), False) # bool tensor
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(encoder_output,
                                          multihead_attention(
                                              query_antecedent=encoder_output,
                                              memory_antecedent=None,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self.hidden_units,
                                              total_value_depth=self.hidden_units,
                                              output_depth=self.hidden_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              name='encoder_self_attention',
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Feed Forward
                encoder_output = residual(encoder_output,
                                          ff_hidden(
                                              inputs=encoder_output,
                                              hidden_size=4 * self.hidden_units,
                                              output_size=self.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=self.residual_dropout_rate)
        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)

        return encoder_output, len_features


class Transformer_Encoder_8x(Transformer_Encoder):

    def encode(self, features, len_sequence):

        encoder_output = tf.layers.dense(
            inputs=features,
            units=self.hidden_units,
            activation=None,
            use_bias=False,
            name='encoder_fc')
        encoder_output = tf.contrib.layers.layer_norm(
            encoder_output, center=True, scale=True, trainable=True)

        # Add positional signal
        encoder_output = add_timing_signal_1d(encoder_output)
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=self.residual_dropout_rate,
                                           training=self.training)
        # Mask
        encoder_padding = tf.equal(tf.sequence_mask(len_sequence, maxlen=tf.shape(features)[1]), False) # bool tensor
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(encoder_output,
                                          multihead_attention(
                                              query_antecedent=encoder_output,
                                              memory_antecedent=None,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self.hidden_units,
                                              total_value_depth=self.hidden_units,
                                              output_depth=self.hidden_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              name='encoder_self_attention',
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Feed Forward
                encoder_output = residual(encoder_output,
                                          ff_hidden(
                                              inputs=encoder_output,
                                              hidden_size=4 * self.hidden_units,
                                              output_size=self.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=self.residual_dropout_rate)

                if i in (1,3,5):
                    encoder_output = tf.layers.max_pooling1d(encoder_output, 2, 2, 'SAME')
                    len_sequence = tf.cast(tf.math.ceil(tf.cast(len_sequence, tf.float32)/2), tf.int32)
                    # Mask
                    encoder_padding = tf.equal(tf.sequence_mask(len_sequence, maxlen=tf.shape(encoder_output)[1]), False)
                    encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)

        return encoder_output, len_sequence


class Conv_Transformer_Encoder(Transformer_Encoder):
    from ..utils.blocks import conv_lstm
    from ..utils.attention import layer_norm
    import numpy as np

    def __init__(self, args, training, name=None):
        super().__init__(args, training, name=None)
        self.num_filters = args.model.encoder.num_filters
        self.feature_map = args.data.num_feat_map

    def __call__(self, features, len_features):

        size_batch  = tf.shape(features)[0]
        size_length = tf.shape(features)[1]
        size_feat = int(self.args.data.dim_input / self.feature_map)
        x = tf.reshape(features, [size_batch, size_length, size_feat, self.feature_map])

        # the first cnn layer
        x = self.normal_conv(
            inputs=x,
            filter_num=self.num_filters,
            kernel=(3,3),
            stride=(2,2),
            padding='SAME',
            use_relu=True,
            name="conv",
            norm_type='layer')
        x = conv_lstm(
            x=x,
            kernel_size=(3,3),
            filters=self.num_filters)

        size_feat = int(np.ceil(size_feat/2)) * self.num_filters
        size_length = tf.cast(tf.math.ceil(tf.cast(size_length, tf.float32)/4), tf.int32)
        len_seq = tf.cast(tf.math.ceil(tf.cast(len_features, tf.float32)/4), tf.int32)
        x = tf.reshape(x, [size_batch, size_length, size_feat])

        encoder_output = tf.layers.dense(
            inputs=x,
            units=self.hidden_units,
            activation=None,
            use_bias=False,
            name='encoder_fc')
        encoder_output = tf.contrib.layers.layer_norm(
            encoder_output, center=True, scale=True, trainable=True)

        # Add positional signal
        encoder_output = add_timing_signal_1d(encoder_output)
        # Dropout
        encoder_output = tf.layers.dropout(encoder_output,
                                           rate=self.residual_dropout_rate,
                                           training=self.training)
        # Mask
        encoder_padding = tf.equal(tf.sequence_mask(len_seq, maxlen=size_length), False) # bool tensor
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("block_{}".format(i)):
                # Multihead Attention
                encoder_output = residual(encoder_output,
                                          multihead_attention(
                                              query_antecedent=encoder_output,
                                              memory_antecedent=None,
                                              bias=encoder_attention_bias,
                                              total_key_depth=self.hidden_units,
                                              total_value_depth=self.hidden_units,
                                              output_depth=self.hidden_units,
                                              num_heads=self.num_heads,
                                              dropout_rate=self.attention_dropout_rate,
                                              name='encoder_self_attention',
                                              summaries=False),
                                          dropout_rate=self.residual_dropout_rate)

                # Feed Forward
                encoder_output = residual(encoder_output,
                                          ff_hidden(
                                              inputs=encoder_output,
                                              hidden_size=4 * self.hidden_units,
                                              output_size=self.hidden_units,
                                              activation=self._ff_activation),
                                          dropout_rate=self.residual_dropout_rate)

                if i in (2, 4):
                    encoder_output = tf.layers.max_pooling1d(encoder_output, 2, 2, 'SAME')

                    size_length = tf.cast(tf.math.ceil(tf.cast(size_length, tf.float32)/2), tf.int32)
                    len_seq = tf.cast(tf.math.ceil(tf.cast(len_seq, tf.float32)/2), tf.int32)
                    encoder_padding = tf.equal(tf.sequence_mask(len_seq, maxlen=size_length), False) # bool tensor
                    encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)

        return encoder_output, len_seq

    def normal_conv(self, inputs, filter_num, kernel, stride, padding, use_relu, name, norm_type="batch"):
        with tf.variable_scope(name):
            net = tf.layers.conv2d(inputs, filter_num, kernel, stride, padding, name="conv")
            if norm_type == "batch":
                net = tf.layers.batch_normalization(net, name="bn")
            elif norm_type == "layer":
                net = layer_norm(net)
            else:
                net = net
            output = tf.nn.relu(net) if use_relu else net

        return output
