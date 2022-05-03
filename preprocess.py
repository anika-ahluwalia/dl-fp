import csv
import os
from typing import Tuple, List
import nltk.stem
from bs4 import BeautifulSoup
import pandas as pd
import string
import re

import numpy as np


def remove_stop_words(raw_string: str) -> str:
    """
    Remove the stop words from a string, based on the NLTK English stop words corpus (includes 179 words).

    :param raw_string: The raw string with stop words.
    :return: The string without stop words.
    """
    tokenizer = nltk.tokenize.ToktokTokenizer()
    tokenized = [token.strip() for token in tokenizer.tokenize(raw_string)]
    stopword_list = nltk.corpus.stopwords.words("english")
    return " ".join(w for w in tokenized if w not in stopword_list)


def remove_html(raw_string: str) -> str:
    """
    Remove HTML tags from text.
    """
    return BeautifulSoup(raw_string, "html.parser").get_text()


def remove_special_chars(raw_string: str) -> str:
    """
    Remove puncuation and special characters.
    """
    return re.sub(r"[^a-zA-z0-9\s]", "", raw_string).translate(str.maketrans("", "", string.punctuation))


def stem(raw_string: str) -> str:
    """
    Stem the string.
    """
    stemmer = nltk.stem.PorterStemmer()
    return " ".join([stemmer.stem(w) for w in raw_string.split()])


def build_vocab_bow(train_inputs, test_inputs):
    # NOTE (lauren): we might need two separate vocabs for train and test -- testing set isn't guaranteed to be a subset of training set
    train_vocab = {}
    test_vocab = {}
    train_unique = np.unique(train_inputs)
    test_unique = np.unique(test_inputs)

    # convert unique words in the training set to unique IDs
    for i in range(len(train_unique)):
        train_vocab[train_unique[i]] = i

    # convert unique words in the testing set to unique IDs
    for i in range(len(test_unique)):
        test_vocab[test_unique[i]] = i

    # NOTE (lauren): the loops convert each of the inputs to their corresponding ids in the vocab!
    # eg) if vocab = {1: "the", 2: "cat"} and train_inputs = ["the", "cat"], then you loop through and convert so that train_inputs = [1, 2]

    # convert the training words
    for i in range(len(train_inputs)):
        train_inputs[i] = train_vocab[train_inputs[i]]

    # convert the test words
    for i in range(len(test_inputs)):
        test_inputs[i] = test_vocab[test_inputs[i]]

    return train_inputs, test_inputs, train_vocab, test_vocab

def get_data(file_path: str, inputs_header: str, labels_header: str) -> Tuple[
        List[str], List[str], List[str], List[str]]:
    """
    Handles processing corpus of IMDb reviews and sentiment labels into training and testing datasets.

    :param file_path: The path to the CSV file containing the IMDb dataset.
    :param inputs_header: The name of the column containing IMDb reviews.
    :param labels_header: The name of the column containing reviews' sentiment labels.
    :return: Four lists: training inputs, training labels, testing inputs, and testing labels.
    """
    dataset: pd.DataFrame = pd.read_csv(file_path)
    raw_inputs: List[str] = dataset[inputs_header].to_list()

    raw_labels = dataset[labels_header].to_list()
    raw_labels = raw_labels.replace('positive', 1)
    raw_labels = raw_labels.replace('negative', 0)
    cleaned_inputs: List[str]
    if not os.path.exists("data/IMDBDataset_cleaned.csv"):
        cleaned_inputs = list(
            map(lambda s: remove_stop_words(stem(remove_special_chars(remove_html(s))).lower()), raw_inputs))

        with open("data/IMDBDataset_cleaned.csv", "w") as cleaned_csv:
            csv_writer = csv.writer(cleaned_csv)
            csv_writer.writerow(["cleaned_review", "sentiment"])
            csv_writer.writerows(list(zip(cleaned_inputs, raw_labels)))
    else:
        cleaned_inputs = list(csv.reader(open("data/IMDBDataset_cleaned.csv", "r")))

    # encode the labels as positive or negative

    # we will split the dataset equally between training and testing
    split_index = len(cleaned_inputs) // 2
    training_inputs = cleaned_inputs[0:split_index + 1]
    training_labels = raw_labels[0:split_index + 1]
    testing_inputs = cleaned_inputs[split_index:]
    testing_labels = raw_labels[split_index:]
    print(training_inputs[:2])
    return training_inputs, training_labels, testing_inputs, testing_labels
