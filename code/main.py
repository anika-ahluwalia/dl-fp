import os
import sys
from typing import List

import nltk
from tqdm import tqdm

from bow_model import BagOfWordsModel
from w2v_sentiment_model import Word2VecSentimentModel
from w2v_model import Word2VecModel
from preprocess import get_data, word2vec_preprocess
import tensorflow as tf
import numpy as np
import gensim


# NOTE (anika): returning list of losses for visualization
def train(model, training_inputs, training_labels):
    losses = []
    for i in tqdm(range(0, len(training_inputs), model.batch_size)):
        batch_inputs = training_inputs[i: i + model.batch_size]
        batch_labels = training_labels[i: i + model.batch_size]

        with tf.GradientTape() as tape:
            predictions = model(batch_inputs)
            loss = model.loss(predictions, batch_labels)
            print(loss)
            losses.append(loss)

        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return losses


def test(model, testing_inputs, testing_labels):
    iterations = int(len(testing_inputs) / model.batch_size)
    accuracy = 0

    for i in tqdm(range(0, len(testing_inputs), model.batch_size)):
        batch_inputs = testing_inputs[i: i + model.batch_size]
        batch_labels = testing_labels[i: i + model.batch_size]
        predictions = model(batch_inputs)
        batch_accuracy = model.accuracy(predictions, batch_labels)
        accuracy = accuracy + batch_accuracy

    print("accuracy", accuracy / iterations)
    return accuracy / iterations


# def visualize_loss(losses):
#     x = [i for i in range(len(losses))]
#     plt.plot(x, losses)
#     plt.title('Loss per batch')
#     plt.xlabel('Batch')
#     plt.ylabel('Loss')
#     plt.show()

def main():
    # check user arguments
    nltk.download("stopwords")
    if len(sys.argv) != 2 or sys.argv[1] not in {"BAG_OF_WORDS", "WORD2VEC", "W2VSENTIMENT"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [BAG_OF_WORDS/WORD2VEC/W2VSENTIMENT]")
        exit()

    # if tfc.remote():
    #     file_path = "gs://dl-fp/data/IMDBDataset.csv"
    #     cleaned_file_path = "gs://dl-fp/data/IMDBDataset_cleaned.csv"
    #     model_save_path = "gs://dl-fp/saved_models/word2vec.ckpt"
    # else:
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
    all_losses = []
    if sys.argv[1] == "BAG_OF_WORDS":
        model = BagOfWordsModel(vocab)
        for epoch in range(num_epochs):
            print("epoch ", epoch)
            losses = train(model, training_inputs, training_labels)
            all_losses = all_losses + losses
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
        words_as_ids = []
        for review in training_inputs:
            review_with_ids = []
            for word in review:
                review_with_ids.append(vocab[word])
            words_as_ids.append(review_with_ids)

        if not os.path.exists("saved_models/word2vec.model"):
            word2vec_model = gensim.models.Word2Vec(sentences=words_as_ids, vector_size=100, window=2, workers=4,
                                                    min_count=1)
            word2vec_model.train(words_as_ids, total_examples=len(words_as_ids), epochs=20)
            word2vec_model.save("saved_models/word2vec.model")
        else:
            word2vec_model = gensim.models.Word2Vec.load("saved_models/word2vec.model")
        model = Word2VecSentimentModel(word2vec_model.wv)

        review_embeddings = []
        for review in words_as_ids:
            word_embeddings = []
            for word in review:
                word_embeddings.append(word2vec_model.wv[word])
            review_embeddings.append(np.average(np.array(word_embeddings), axis=0))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss='binary_crossentropy',
            metrics=["accuracy"]
        )
        model.fit(
            x=np.array(review_embeddings),
            y=np.array(training_labels),
            epochs=40,
            batch_size=120
        )

    # if (len(all_losses) > 0):
    #     visualize_loss(all_losses)

    print("testing...")
    test(model, testing_inputs, testing_labels)


if __name__ == '__main__':
    main()
