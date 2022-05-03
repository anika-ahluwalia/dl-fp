import tensorflow as tf
import numpy as np
from tensorflow.keras import Model

class Word2VecModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size):
        super(Word2VecModel, self).__init__()

        self.batch_size = 120

        self.learning_rate = 0.01
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        #COPIED FROM WORD2VEC LAB BY NAOMI
        self.embedding = tf.keras.layers.Embedding(vocab_size, vocab_size, input_length=vocab_size)
        self.mlp = tf.keras.layers.Dense(vocab_size) #multi-layered perceptron!
    
    @tf.function
    # inputs is a list of size batch size that has a lists of all of the words in the reviews
    def call(self, inputs):
        # TODO: Naomi and Galen fill in Word2Vec
        print("Word2Vec")
        embedding = self.embedding(inputs)
        logits = self.mlp(embedding)
        return tf.nn.softmax(logits)

    # hw 3 
    def loss(self, logits, labels):
        prob = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        correct_predictions = tf.equal(tf.argmax(predictions, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))