import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from sklearn.feature_extraction.text import CountVectorizer

class BagOfWordsModel(tf.keras.Model):
    def __init__(self):
        super(BagOfWordsModel, self).__init__()

        self.batch_size = 120

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        self.vectorizer = CountVectorizer()

    # def create_bag_of_words(inputs):
    #     vectors = []
    #     for review in inputs:
    #         review_words = review.split()
    #         word_occurrences = {}
    #         for word in review_words:
    #             if word in word_occurrences:
    #                 word_occurrences[word] = word_occurrences.get(word) + 1
    #             else:
    #                 word_occurrences[word] = 1

    #         vectorized_words = [0] * len(review_words)
    #         for i in range(len(review_words)):
    #             vectorized_words[i] = word_occurrences[review_words[i]]
    #         vectors.append(vectorized_words)
    #     return vectors

    def create_bag_of_words(self, inputs):
        bag = self.vectorizer.fit_transform(inputs)
        return bag.toarray()
        
    
    @tf.function
    # inputs is a list of size batch size that has a lists of all of the words in the reviews
    def call(self, inputs):
        bag = self.create_bag_of_words(inputs)
        print(bag)

    # hw 3 
    def loss(self, logits, labels):
        prob = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))