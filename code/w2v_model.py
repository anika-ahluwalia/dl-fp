import numpy as np
import tensorflow as tf


# Largely copied from word2vec lab
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

        self.batch_size = 120

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.network = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size, embedding_size),
            tf.keras.layers.Dense(vocab_size)
        ])

    def call(self, inputs):
        """
        Generates a probability distribution over all words in the corpus for which word likely follows the current word.

        :param inputs: A list of size batch_size that
        """
        return self.network(inputs)

    # def loss(self, logits, labels):
    #     logits = tf.reshape(logits, [-1])
    #     prob = tf.keras.losses.binary_crossentropy(labels, logits)
    #     loss = tf.reduce_mean(tf.cast(prob, tf.float32))
    #     return loss
    #
    # def accuracy(self, predictions, labels):
    #     predictions = tf.reshape(predictions, [-1])
    #     correct_predictions = tf.equal(tf.argmax(predictions), tf.argmax(labels))
    #     return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
