import tensorflow as tf
import numpy as np

class BagOfWordsModel(tf.keras.Model):
    def __init__(self, vocab):
        super(BagOfWordsModel, self).__init__()

        self.batch_size = 100
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_size = 200

        self.hidden_layer_size = 100

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.max_len = 1  # this is set after we pad the inputs -- see note on line 83

        # NOTE (lauren): lol one more bug -- all of the logits have a value of 1.??
        self.network = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.max_len),
            tf.keras.layers.LSTM(self.embedding_size),
            tf.keras.layers.Dense(250, activation='relu'),
            tf.keras.layers.Dense(self.hidden_layer_size, activation='relu'),
            tf.keras.layers.Dense(1, activation='softmax')
        ])

    def create_bag_of_words(self, inputs):
        bag = []
        review_length = []
        for review in inputs:
            vector = [0] * len(self.vocab)
            review_length.append(len(review))
            for word in review:
                number = self.vocab[word]
                vector[number] += 1
            bag.append(vector)
        self.max_len = int(np.ceil(np.mean(review_length)))
        return bag

    def pad_bags(self, vectorized_inputs):
        return tf.keras.preprocessing.sequence.pad_sequences(vectorized_inputs, maxlen=self.max_len, padding='post', truncating='post')

    def call(self, inputs):
        bag = self.create_bag_of_words(inputs)
        padded_inputs = self.pad_bags(bag)
        logits = self.network(padded_inputs)
        return logits

    def loss(self, logits, labels):
        logits = tf.reshape(logits, [-1])
        prob = tf.keras.losses.binary_crossentropy(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        predictions = tf.reshape(predictions, [-1])
        correct_predictions = tf.equal(tf.argmax(predictions), tf.argmax(labels))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))