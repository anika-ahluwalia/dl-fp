import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class AnalysisModel(tf.keras.Model):
    def __init__(self):
        super(AnalysisModel, self).__init__()

        self.batch_size = 120

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

    
    @tf.function
    def call(self):
        pass

    # hw 3 
    def loss(self, logits, labels):
        prob = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))