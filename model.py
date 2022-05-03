import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class AnalysisModel(tf.keras.Model):
    def __init__(self, is_bow):
        super(AnalysisModel, self).__init__()
        self.is_bow = is_bow
    
    @tf.function
    def call(self):
        if self.is_bow:
            print("is_bow is true in model.py")
        else:
            print("is_bow is false in model.py")

    # hw 3 
    def loss(self, logits, labels):
        prob = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))