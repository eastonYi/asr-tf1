from abc import ABCMeta, abstractmethod
import tensorflow as tf


class Encoder(object):
    __metaclass__ = ABCMeta

    def __init__(self, args, training, name=None):
        '''EDEncoder constructor

        Args:
            args: the encoder configuration
            name: the encoder name
            constraint: the constraint for the variables
        '''
        self.args = args
        self.name = name
        self.training = training

    @abstractmethod
    def __call__(self, features, len_features):
        '''
        Create the variables and do the forward computation

        Args:
            inputs: the inputs to the neural network, this is a dictionary of
                [batch_size x time x ...] tensors
            input_seq_length: The sequence lengths of the input utterances, this
                is a dictionary of [batch_size] vectors
            training: whether or not the network is in training mode

        Returns:
            - the outputs of the encoder as a dictionary of
                [bath_size x time x ...] tensors
            - the sequence lengths of the outputs as a dictionary of
                [batch_size] tensors
        '''

    @property
    def variables(self):
        '''get a list of the models's variables'''
        variables = tf.get_collection(
            tf.GraphKeys.GLOBAL_VARIABLES,
            scope=self.name + '/')

        if hasattr(self, 'wrapped'):
            #pylint: disable=E1101
            variables += self.wrapped.variables

        return variables
