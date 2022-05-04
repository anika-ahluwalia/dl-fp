import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class BagOfWordsModel(tf.keras.Model):
    def __init__(self, vocab):
        super(BagOfWordsModel, self).__init__()

        self.batch_size = 120
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_size = 200

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.max_len = 0  # this is set after we pad the inputs -- see note on line 83

        # NOTE (lauren): do we need an LSTM? some models online use it, others don't
        ## (anika): I think that we do for RNN
        self.GRU = tf.keras.layers.GRU(self.batch_size, return_sequences=True, return_state=True)

        self.network = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.max_len),
            tf.keras.layers.Flatten(),

            # NOTE (lauren): something is up with the data shapes -- commenting out while i look at train/test
            # tf.keras.layers.Dense(250, activation=None),
            # tf.keras.layers.Dense(1, activation='softmax'),  # should output one value
        ])

    def create_bag_of_words(self, inputs):
        bag = []
        for review in inputs:
            vector = [0] * len(self.vocab)
            for word in review:
                number = self.vocab[word]
                vector[number] += 1
            bag.append(vector)
        return bag

    # input: vectorized_inputs -- a vectorized array of [batch_size, ~length of each review~]
    # output: an array of [batch_size, max_len], where we pad review rows that are not of length max_len
    def pad_bags(self, vectorized_inputs):
        return tf.keras.preprocessing.sequence.pad_sequences(vectorized_inputs, padding='post')
    
    @tf.function
    def call(self, inputs):
        bags = self.create_bag_of_words(inputs)
        padded_inputs = self.pad_bags(bags)
        
        # set max_len -- to be used in the embedding layer above
        self.max_len = len(padded_inputs[0])
        logits = self.network(padded_inputs)

        return logits

    def loss(self, logits, labels):
        prob = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))