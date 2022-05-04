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

# param inputs: takes in a list of cleaned inputs
# output: a vocab of the unique input words
def build_vocab(inputs):
    vocab = {}

    all_words = []

    for review in inputs:
        for word in review:
            all_words.append(word)
    
    unique_inputs = np.unique(all_words)

    # convert unique input words to unique IDs
    for i in range(len(unique_inputs)):
        vocab[unique_inputs[i]] = i

    # NOTE (lauren): the loops convert each of the inputs to their corresponding ids in the vocab!
    # eg) if vocab = {1: "the", 2: "cat"} and train_inputs = ["the", "cat"], then you loop through and convert so that train_inputs = [1, 2]
    # convert the training words
    # for i in range(len(train_inputs)):
    #     train_inputs[i] = train_vocab[train_inputs[i]]

    # # convert the test words
    # for i in range(len(test_inputs)):
    #     test_inputs[i] = test_vocab[test_inputs[i]]

    # NOTE (anika) ^ i think we need to preserve inputs as is for now so commented out

    return vocab

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

    # getting rid of headers
    raw_inputs = raw_inputs[1:]
    raw_labels = raw_labels[1:]
    

    # encode the labels as positive or negative
    cleaned_labels = []
    for label in raw_labels:
        if label == 'positive':
            cleaned_labels.append(1)
        elif label == 'negative':
            cleaned_labels.append(0)

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
        cleaned_inputs = cleaned_inputs[1:]

    ready_inputs = []
    for review in cleaned_inputs:
        review_as_list = review[0].split()
        ready_inputs.append(review_as_list)
    
    # build vocab
    vocab = build_vocab(ready_inputs)

    # we will split the dataset equally between training and testing
    split_index = (len(ready_inputs) // 10) * 7  # changed to 70/30 split
    training_inputs = ready_inputs[0:split_index + 1]
    training_labels = cleaned_labels[0:split_index + 1]
    testing_inputs = ready_inputs[split_index:]
    testing_labels = cleaned_labels[split_index:]

    
    return training_inputs, training_labels, testing_inputs, testing_labels, vocab
