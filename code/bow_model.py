import tensorflow as tf
import numpy as np
from keras import regularizers

class BagOfWordsModel(tf.keras.Model):
    def __init__(self, vocab):
        """
        The BagOfWordsModel class predicts sentiment of a review as either positive or negative.

        :param vocab: A dictionary mapping all words used in reviews to unique IDs.
        """
        super(BagOfWordsModel, self).__init__()

        # initializing hyper-parameters
        self.batch_size = 100
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding_size = 200
        self.hidden_layer_size = 100
        self.learning_rate = 0.0001

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # creating a network of 3 dense layers (with a hidden layer)
        # using sigmoid activation function at the end
        self.network = tf.keras.Sequential([
            tf.keras.layers.Dense(self.batch_size, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal',  kernel_regularizer=regularizers.l1(0.001)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(self.hidden_layer_size, activation='relu', kernel_initializer='random_normal', bias_initializer='random_normal',  kernel_regularizer=regularizers.l1(0.001)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, kernel_initializer='random_normal', bias_initializer='random_normal', activation='sigmoid',  kernel_regularizer=regularizers.l1(0.001))
        ])

    def create_bag_of_words(self, inputs):
        """
        Method to create the bag of words (that is passed through the model)
        Bag of Words consists of a vector for each review in which the vector is the
        length of the vocabulary. Each index of the vector represents the prevalence
        of that vocabulary word in the review.
        
        :param inputs: the cleaned list of reviews (as lists of words)
        :return: the bag of words (list of the vectors of each review)
        """
        bag = []
        review_length = []
        # creating a vector for each review
        for review in inputs:
            vector = [0] * len(self.vocab)
            review_length.append(len(review))
            # incrementing indices for each word present in the review
            for word in review:
                number = self.vocab[word]
                vector[number] += 1
            bag.append(vector)
        self.max_len = int(np.ceil(np.mean(review_length)))
        return bag

    def call(self, inputs):
        """
        Call function for the model. Generates predictions of sentiment for reviews.

        :param inputs: the cleaned list of reviews (as lists of words)
        :return: a list of the probabilities generated for each review.
        """
        bag = self.create_bag_of_words(inputs)
        bag = tf.convert_to_tensor(bag)
        logits = self.network(bag)
        logits = tf.reshape(logits, (len(inputs),))
        return logits

    def loss(self, probabilities, labels):
        """
        Calculating loss of the model.

        :param probabilities: a list of the probabilities generated for each review
        :param labels: a list of the correct labels for each review
        :return: the loss of this batch of predictions
        """
        # using binary cross entropy loss for binary classification
        prob = tf.keras.losses.binary_crossentropy(tf.convert_to_tensor(labels, dtype=tf.float32), probabilities, from_logits=False)
        loss = tf.reduce_mean(tf.cast(prob, tf.float32))
        return loss

    def accuracy(self, predictions, labels):
        """
        Calculating accuracy of the model.

        :param probabilities: a list of the probabilities generated for each review
        :param labels: a list of the correct labels for each review
        :return: the accuracy of this batch of predictions
        """
        # translating the predictions (probabilities) to be in terms of labels
        classify = lambda prob: 1 if prob >= 0.5 else 0
        classifications = tf.map_fn(classify, predictions)
        # calculating the percentage of correct predictions
        correct_predictions = tf.equal(classifications, tf.convert_to_tensor(labels, dtype=tf.float32))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))