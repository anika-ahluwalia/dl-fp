import sys

from tqdm import tqdm

from bow_model import BagOfWordsModel
from w2v_model import Word2VecModel
from preprocess import get_data, word2vec_preprocess
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

# NOTE (anika): returning list of losses for visualization
def train(model, training_inputs, training_labels):
    losses = []
    for i in tqdm(range(0, len(training_inputs), model.batch_size)):
        batch_inputs = training_inputs[i: i + model.batch_size]
        batch_labels = training_labels[i: i + model.batch_size]

        with tf.GradientTape() as tape:
            predictions = model(batch_inputs)
            loss = model.loss(predictions, batch_labels)
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

def visualize_loss(losses): 
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()

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

    print("preprocessing the data...")
    training_inputs, training_labels, testing_inputs, testing_labels, vocab = get_data(file_path, input_header,
                                                                                       label_header)

    # initialize model as bag of words or word2vec
    print("making the model...")
    if sys.argv[1] == "BAG_OF_WORDS":
        model = BagOfWordsModel(vocab)
    elif sys.argv[1] == "WORD2VEC":
        training_inputs = word2vec_preprocess(training_inputs, vocab, 2)
        testing_inputs = word2vec_preprocess(testing_inputs, vocab, 2)
        model = Word2VecModel(len(vocab), 100)

    # train and test data
    print("training the model...")
    all_losses = []
    if not sys.argv[1] == "WORD2VEC":
        for epoch in range(num_epochs):
            print("epoch ", epoch)
            losses = train(model, training_inputs, training_labels)
            all_losses = all_losses + losses
    else:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="saved_models/word2vec.ckpt",
                                                         save_weights_only=True,
                                                         verbose=1)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.nn.sparse_softmax_cross_entropy_with_logits,
            metrics=[]
        )
        model.fit(
            np.array(training_inputs)[:, [0]],
            np.array(training_inputs)[:, [1]],
            epochs=20,
            batch_size=120,
            callbacks=[cp_callback]
        )
    
    if (len(all_losses) > 0):
        visualize_loss(all_losses)

    print("testing...")
    test(model, testing_inputs, testing_labels)


if __name__ == '__main__':
    main()
