import tensorflow as tf
from .encoder import Encoder
from ..utils.attention import residual, multihead_attention, ff_hidden,\
    attention_bias_ignore_padding, add_timing_signal_1d


class Transformer_Encoder(Encoder):
    def __init__(self, args, training, embed_table=None, name=None):
        super().__init__(args, training, embed_table=None, name=None)
        self.attention_dropout_rate = args.model.encoder.attention_dropout_rate if training else 0.0
        self.residual_dropout_rate = args.model.encoder.residual_dropout_rate if training else 0.0
        self.hidden_units = args.model.encoder.num_cell_units
        self.num_heads = args.model.encoder.num_heads
        self.num_blocks = args.model.encoder.num_blocks
        self._ff_activation = lambda x, y: x * tf.sigmoid(y)

    def encode(self, features, len_features):

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
