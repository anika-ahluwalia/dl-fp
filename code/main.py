import os
import sys
from typing import List
from matplotlib import pyplot as plt

import nltk
from tqdm import tqdm

from bow_model import BagOfWordsModel
from w2v_sentiment_model import Word2VecSentimentModel
from w2v_model import Word2VecModel
from preprocess import get_data, word2vec_preprocess, words_to_ids, word2vec_sentiment_preprocess
import tensorflow as tf
import numpy as np
import gensim


def train(model, training_inputs, training_labels):
    """
    Method to train the model (either model) in batched inputs over all training data.

    :param model: the model upon which we are training
    :param training_inputs: list of the preprocessed training reviews
    :param training_labels: list of the labels associated with training reviews
    :return: list of the accuracies over all training batches
    :return: list of the losses over all training batches
    """
    # lists to store loss and accuracy
    losses = []
    accuracies = []

    for i in tqdm(range(0, len(training_inputs), model.batch_size)):
        # batching inputs
        batch_inputs = training_inputs[i: i + model.batch_size]
        batch_labels = training_labels[i: i + model.batch_size]

        # ensuring that we always have complete batches
        if len(batch_inputs) == model.batch_size:
            with tf.GradientTape() as tape:
                # generating predictions
                predictions = model(batch_inputs)
                # storing loss and accuracy
                loss = model.loss(predictions, batch_labels)
                losses.append(loss)
                accuracy = model.accuracy(predictions, batch_labels)
                accuracies.append(accuracy)

            gradients = tape.gradient(loss, model.trainable_variables)
            model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses, accuracies


def test(model, testing_inputs, testing_labels):
    """
    Method to test the model (either model) in batched inputs on testing data.

    :param model: the model that we are testing.
    :param testing_inputs: list of the preprocessed testing reviews
    :param testing_labels: list of the labels associated with testing reviews
    :return: the overall accuracy of the model on the testing data
    """
    iterations = int(len(testing_inputs) / model.batch_size)
    accuracy = 0

    for i in tqdm(range(0, len(testing_inputs), model.batch_size)):
        # batching inputs
        batch_inputs = testing_inputs[i: i + model.batch_size]
        batch_labels = testing_labels[i: i + model.batch_size]
        if len(batch_inputs) == model.batch_size:
            predictions = model(batch_inputs)
            batch_accuracy = model.accuracy(predictions, batch_labels)
            accuracy = accuracy + batch_accuracy

    print("accuracy", accuracy / iterations)
    return accuracy / iterations


def visualize_loss(losses):
    """
    Method to visualize the loss of the model over training.

    :param losses: a list of the losses in each training batch
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

def visualize_accuracy(accuracies):
    """
    Method to visualize the accuracy of the model over training.

    :param losses: a list of the accuracies in each training batch
    """
    x = [i for i in range(len(accuracies))]
    plt.plot(x, accuracies)
    plt.title('Accuracy per batch')
    plt.xlabel('Batch')
    plt.ylabel('Accuracy')
    plt.show()

def main():
    # checking user arguments
    nltk.download("stopwords")
    if len(sys.argv) != 2 or sys.argv[1] not in {"BAG_OF_WORDS", "WORD2VEC", "W2VSENTIMENT"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [BAG_OF_WORDS/WORD2VEC/W2VSENTIMENT]")
        exit()

    # storing important variables - DO NOT CHANGE
    file_path = "data/IMDBDataset.csv"
    cleaned_file_path = "data/IMDBDataset_cleaned.csv"
    model_save_path = "saved_models/word2vec.ckpt"
    input_header = "review"
    label_header = "sentiment"
    num_epochs = 1

    print("preprocessing the data...")
    training_inputs, training_labels, testing_inputs, testing_labels, vocab = get_data(file_path, cleaned_file_path,
                                                                                       input_header, label_header)

    # initialize model as bag of words or word2vec
    print("making the model...")

    # lists to store loss and accuracy for visualization
    all_losses = []
    all_accuracies = []

    if sys.argv[1] == "BAG_OF_WORDS":
        model = BagOfWordsModel(vocab)
        for epoch in range(num_epochs):
            print("epoch ", epoch)
            losses, accuracies = train(model, training_inputs, training_labels)
            all_losses = all_losses + losses
            all_accuracies = all_accuracies + accuracies
    elif sys.argv[1] == "WORD2VEC":
        training_inputs = word2vec_preprocess(training_inputs, vocab, 2)
        testing_inputs = word2vec_preprocess(testing_inputs, vocab, 2)
        model = Word2VecModel(len(vocab), 100)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_save_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits,
            metrics=[]
        )
        model.fit(
            x=np.array(training_inputs)[:, 0].reshape((len(training_inputs), 1)),
            y=np.array(training_inputs)[:, 1],
            epochs=20,
            batch_size=120,
            callbacks=[cp_callback]
        )
    elif sys.argv[1] == "W2VSENTIMENT":
        training_words_as_ids: list[list[int]] = words_to_ids(training_inputs, vocab)
        testing_words_as_ids: list[list[int]] = words_to_ids(testing_inputs, vocab)

        if not os.path.exists("saved_models/word2vec.model"):
            word2vec_model = gensim.models.Word2Vec(sentences=training_words_as_ids + testing_words_as_ids,
                                                    vector_size=100, window=2, workers=4,
                                                    min_count=1)
            word2vec_model.train(training_words_as_ids + testing_words_as_ids,
                                 total_examples=len(training_words_as_ids) + len(testing_words_as_ids), epochs=20)
            word2vec_model.save("saved_models/word2vec.model")
        else:
            word2vec_model = gensim.models.Word2Vec.load("saved_models/word2vec.model")
        model = Word2VecSentimentModel(word2vec_model.wv)

        training_review_embeddings = word2vec_sentiment_preprocess(training_words_as_ids, word2vec_model)
        testing_review_embeddings = word2vec_sentiment_preprocess(testing_words_as_ids, word2vec_model)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=["accuracy"]
        )
        model.fit(
            x=training_review_embeddings,
            y=np.array(training_labels),
            epochs=40,
            batch_size=120
        )
        model.evaluate(
            x=testing_review_embeddings,
            y=np.array(testing_labels)
        )

    # if lists were populated, visualize the loss and accuracy
    if (len(all_accuracies) > 0):
        visualize_accuracy(all_accuracies)
    if (len(all_losses) > 0):
        visualize_loss(all_losses)

    print("testing...")
    test(model, testing_inputs, testing_labels)


if __name__ == '__main__':
    main()
