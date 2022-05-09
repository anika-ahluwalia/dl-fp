import tensorflow as tf
import numpy as np
# NOTE (anika): added because of addition on line 27
import keras.backend as K
from keras import regularizers

class BagOfWordsModel(tf.keras.Model):
    def __init__(self, vocab):
        super(BagOfWordsModel, self).__init__()

        self.batch_size = 100
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_size = 200

        self.hidden_layer_size = 100

        self.learning_rate = 0.0001
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.max_len = 1  # this is set after we pad the inputs -- see note on line 83

        self.network = tf.keras.Sequential([
            # tf.keras.layers.Embedding(self.vocab_size, self.embedding_size, input_length=self.max_len),
            # NOTE (anika): found an architecture online that doesn't use LSTM -- instead uses Lambda with mean so decided to try it out
            # https://analyticsindiamag.com/the-continuous-bag-of-words-cbow-model-in-nlp-hands-on-implementation-with-codes/
            # tf.keras.layers.LSTM(self.embedding_size),
            # tf.keras.layers.Lambda(lambda x: K.mean(x, axis=1), output_shape=(self.batch_size,)),
            # tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.batch_size, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal',  kernel_regularizer=regularizers.l1(0.001)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.hidden_layer_size, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal',  kernel_regularizer=regularizers.l1(0.001)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, kernel_initializer='random_normal', bias_initializer='random_normal', activation='sigmoid',  kernel_regularizer=regularizers.l1(0.001))
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
        # padded_inputs = self.pad_bags(bag)
        bag = tf.convert_to_tensor(bag)
        logits = self.network(bag)
        logits = tf.reshape(logits, (self.batch_size,))
        return logits

    def loss(self, probabilities, labels):
        prob = tf.keras.losses.binary_crossentropy(tf.convert_to_tensor(labels, dtype=tf.float32), probabilities, from_logits=False)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):

        # NOTE (anika): trying to make our accuracy mean something
        # right now its the probabilities and strict equality
        classify = lambda prob: 1 if prob >= 0.5 else 0
        classifications = tf.map_fn(classify, predictions)

        # correct_predictions = tf.equal(tf.argmax(predictions), tf.argmax(tf.convert_to_tensor(labels, dtype=tf.float32)))
        correct_predictions = tf.equal(classifications, tf.convert_to_tensor(labels, dtype=tf.float32))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))