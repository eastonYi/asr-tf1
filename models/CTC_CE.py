import tensorflow as tf
import logging
import sys
import math

from .utils.regularization import confidence_penalty
from .utils.blocks import shrink_layer
from .utils.tools import dense_sequence_to_sparse, choose_device, smoothing_cross_entropy
from .Ectc_Docd import Ectc_Docd
from .decoders.fc_decoder import FCDecoder
from .utils.tfAudioTools import batch_splice

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(levelname)s(%(filename)s:%(lineno)d): %(message)s')


class CTC_CE(Ectc_Docd):
    '''
    CTC model is viewed as seq2seq model with the final FC layer as decoder.
    '''
    def __init__(self, tensor_global_step, encoder, decoder, training, args,
                 batch=None, name='CTC_CE'):
        super().__init__(tensor_global_step, encoder, decoder, training, args, batch, name)

    def __call__(self, feature, len_features, reuse=False):
        with tf.variable_scope(self.name, reuse=reuse):
            encoder_ctc = self.gen_encoder(
                training=self.training,
                args=self.args,
                name='ctc_encoder')
            decoder_ctc = FCDecoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args,
                name='ctc_decoder')
            decoder_ce = self.gen_decoder(
                training=self.training,
                global_step=self.global_step,
                args=self.args,
                name='ce_model')

            with tf.variable_scope(encoder_ctc.name or 'encoder_ctc'):
                encoded, len_encoded = encoder_ctc(feature, len_features)

            with tf.variable_scope(decoder_ctc.name or 'decoder_ctc'):
                logits_ctc, align, len_logits_ctc = decoder_ctc(
                    encoded, len_encoded, None,
                    shrink=False,
                    num_fc=self.args.model.decoder.num_fc,
                    hidden_size=self.args.model.decoder.hidden_size,
                    dim_output=self.args.dim_output)

            feature_splice = batch_splice(feature, 2, 1)

            # shrink layer
            encoded_ce, len_encoded_ce = shrink_layer(
                feature_splice, len_encoded, logits_ctc,
                feature_splice.get_shape()[-1])

            with tf.variable_scope(decoder_ce.name or 'decoder_ce'):
                logits_ce, decoded, len_logits_ce = decoder_ce(encoded_ce, len_encoded_ce, None)

        return [logits_ctc, logits_ce], [align, decoded], [len_logits_ctc, len_logits_ce]
