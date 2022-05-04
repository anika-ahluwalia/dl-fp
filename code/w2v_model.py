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

        self.batch_size = 120

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # COPIED FROM WORD2VEC LAB BY NAOMI
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_size)
        # Multi-layered perceptron!
        self.mlp = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def call(self, inputs):
        """
        Generates a probability distribution over all words in the corpus for which word likely follows the current word.

        :param inputs: A list of size batch_size that
        """

        # error: got shape [120] but wanted [120, 91]
        # should we do something to inputs before passing them in?
        embedding = self.embedding(tf.convert_to_tensor(inputs))
        return self.mlp(embedding)

    def loss(self, logits, labels):
        prob = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
