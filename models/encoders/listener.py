'''@file listener.py
contains the listener code'''

import tensorflow as tf
from .encoder import Encoder

from nabu.neuralnetworks.components import layer


class Listener(Encoder):
    '''a listener object
    transforms input features into a high level representation'''

    def encode(self, features, len_feas):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: [batch_size x time x ...] tensor
            input_seq_length: [batch_size] vector
            is_train: whether or not the network is in training mode

        Returns:
            - [bath_size x time x ...] tensor
            - [batch_size] tensor
        '''
        num_pblayers = self.args.model.encoder.num_pblayers
        num_blayers = self.args.model.encoder.num_blayers
        num_cell_units = self.args.model.encoder.num_cell_units
        dropout = self.args.model.encoder.dropout

        outputs = features
        output_seq_lengths = len_feas
        for l in range(num_pblayers):
            outputs, output_seq_lengths = layer.pblstm(
                inputs=outputs,
                sequence_length=output_seq_lengths,
                num_units=num_cell_units,
                num_steps=2,
                layer_norm=True,
                scope='en_pblstm_%d' % l)

            if dropout > 0 and self.is_train:
                outputs = tf.nn.dropout(outputs, keep_prob=1.0-dropout)

        for l in range(num_blayers):
            outputs = layer.blstm(
                inputs=outputs,
                sequence_length=output_seq_lengths,
                num_units=num_cell_units,
                scope='en_blstm_%d' % (l+num_pblayers))

            if dropout > 0 and self.is_train:
                outputs = tf.nn.dropout(outputs, keep_prob=1.0-dropout)

        return outputs, output_seq_lengths
