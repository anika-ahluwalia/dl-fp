import sys
from bow_model import BagOfWordsModel
from w2v_model import Word2VecModel
from preprocess import get_data, build_vocab_bow
import tensorflow as tf
import numpy as np


# from hw 2
def train(model, training_inputs, training_labels):
    random_indices = tf.random.uniform([len(training_inputs)], minval=0, maxval=len(training_inputs) - 1, dtype=tf.dtypes.int32)
    shuffled_indices = tf.random.shuffle(random_indices)
    shuffled_inputs = tf.gather(training_inputs, shuffled_indices)
    shuffled_labels = tf.gather(training_labels, shuffled_indices)

    iterations = int(len(training_inputs) / model.batch_size)

    inputs = tf.split(shuffled_inputs, iterations)
    labels = tf.split(shuffled_labels, iterations)

    # batching in here
    for (batch_inputs, batch_labels) in zip(inputs, labels):
        batch_inputs = tf.image.random_flip_left_right(batch_inputs)

        with tf.GradientTape() as tape:
            predictions = model(batch_inputs)
            loss = model.loss(predictions, batch_labels)
    
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# from hw 2
def test(model, testing_inputs, testing_labels):
    iterations = int(len(testing_inputs) / model.batch_size)
    inputs = tf.split(testing_inputs, iterations)
    labels = tf.split(testing_labels, iterations)
    accuracy = 0
    for (batch_input, batch_label) in zip(inputs, labels):
        predictions = model(batch_input, True)
        batch_accuracy = model.accuracy(predictions, batch_label)
        accuracy = accuracy + batch_accuracy

    print("accuracy", accuracy / iterations)
    return accuracy / iterations


def main():
    # check user arguments
    if len(sys.argv) != 2 or sys.argv[1] not in {"BAG_OF_WORDS", "WORD2VEC"}:
        print("USAGE: python main.py <Model Type>")
        print("<Model Type>: [BAG_OF_WORDS/WORD2VEC]")
        exit()

    file_path = 'data/IMDBDataset.csv'
    input_header = "review"
    label_header = "sentiment"
    num_epochs = 1  # NOTE (lauren): changed from 50 epochs to 1 (for now) so we don't overfit

    # NOTE (lauren): what if we use this instead??
    # (X_train, y_train), (X_test, y_test) = tf.keras.datasets.imdb.load_data()
    # X = np.concatenate((X_train, X_test), axis=0)
    # y = np.concatenate((y_train, y_test), axis=0)
    # print(X.shape)

    training_inputs, training_labels, testing_inputs, testing_labels = get_data(file_path, input_header, label_header)
    vocab = build_vocab_bow(training_inputs, testing_inputs)

    # initialize model as bag of words or word2vec
    if sys.argv[1] == "BAG_OF_WORDS":
        model = BagOfWordsModel(vocab)
    elif sys.argv[1] == "WORD2VEC":
        model = Word2VecModel()

    # train and test data
    for epoch in range(num_epochs):
        print("epoch ", epoch)
        train(model, training_inputs, training_labels)
    test(model, testing_inputs, testing_labels)


if __name__ == '__main__':
    main()
