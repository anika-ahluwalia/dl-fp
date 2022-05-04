import sys
from bow_model import BagOfWordsModel
from w2v_model import Word2VecModel
from preprocess import get_data
import tensorflow as tf
import numpy as np

def train(model, training_inputs, training_labels):

    for i in range(0, len(training_inputs), model.batch_size):
        batch_inputs = training_inputs[i : i + model.batch_size]
        batch_labels = training_labels[i : i + model.batch_size]

        with tf.GradientTape() as tape:
            predictions = model(batch_inputs)
            loss = model.loss(predictions, batch_labels)
    
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def test(model, testing_inputs, testing_labels):
    iterations = int(len(testing_inputs) / model.batch_size)
    accuracy = 0
    for i in range(0, len(testing_inputs), model.batch_size):
        batch_inputs = testing_inputs[i : i + model.batch_size]
        batch_labels = testing_labels[i : i + model.batch_size]
        predictions = model(batch_inputs, True)
        batch_accuracy = model.accuracy(predictions, batch_labels)
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
    num_epochs = 1

    training_inputs, training_labels, testing_inputs, testing_labels, vocab = get_data(file_path, input_header, label_header)

    # initialize model as bag of words or word2vec
    if sys.argv[1] == "BAG_OF_WORDS":
        model = BagOfWordsModel(vocab)
    elif sys.argv[1] == "WORD2VEC":
        model = Word2VecModel(len(vocab), 100)

    # train and test data
    for epoch in range(num_epochs):
        print("epoch ", epoch)
        train(model, training_inputs, training_labels)
    test(model, testing_inputs, testing_labels)


if __name__ == '__main__':
    main()
