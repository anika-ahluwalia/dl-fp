import numpy as np
import tensorflow as tf


class Word2VecModel(tf.keras.Model):
    """
    Model for generating vector representations of words in a corpus.
    """

    def __init__(self, vocab_size, embedding_size):
        """
        Instantiates the model with an embedding layer and a fully connected layer.

        :param vocab_size: The number of unique words in the corpus.
        :param embedding_size: The cardinality of the embedding vectors the model should produce.
        """
        super(Word2VecModel, self).__init__()

        self.network = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_size),
            tf.keras.layers.Dense(vocab_size)
        ])

    def call(self, inputs):
        """
        Generates a probability distribution over all words in the corpus for which word likely follows the current word.

        :param inputs: A list of size batch_size that
        """
        return tf.squeeze(self.network(inputs))