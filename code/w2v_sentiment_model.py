from typing import List

import tensorflow as tf
import numpy as np

class Word2VecSentimentModel(tf.keras.Model):
    def __init__(self, embeddings):
        super(Word2VecSentimentModel, self).__init__()

        self.embeddings = embeddings
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dense(50, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax")
        ])

    def call(self, inputs: List[List[int]]):
        review_embeddings = []
        for review in inputs:
            word_embeddings = []
            for word in review:
                word_embeddings.append(self.embeddings[word])
            review_embeddings.append(np.average(np.array(word_embeddings, axis=0)))

        return self.network(tf.convert_to_tensor(review_embeddings))
