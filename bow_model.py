import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class BagOfWordsModel(tf.keras.Model):
    def __init__(self):
        super(BagOfWordsModel, self).__init__()

        self.batch_size = 120

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    
    def create_bag_of_words(sentence_list):
        word_occurrences = {}
        for word in sentence_list:
            if word in word_occurrences:
                word_occurrences[word] = word_occurrences.get(word) + 1
            else:
                word_occurrences[word] = 1

        vectorized_words = sentence_list
        for word in vectorized_words:
            print(word)
    
    @tf.function
    # inputs is a list of size batch size that has a lists of all of the words in the reviews
    def call(self, inputs):
        print(inputs)

    # hw 3 
    def loss(self, logits, labels):
        prob = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))