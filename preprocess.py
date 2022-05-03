from typing import Tuple, List

import nltk.stem
from bs4 import BeautifulSoup

import pandas as pd
import string
import re


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
    vocab = {}
    unique = np.unique(train_inputs)
    # convert to unique IDs
    for i in range(len(unique)):
        vocab[unique[i]] = i

    # convert the training words
    for i in range(len(train_inputs)):
        train_inputs[i] = vocab[train_inputs[i]]

    # convert the test words
    for i in range(len(test_inputs)):
        test_inputs[i] = vocab[test_inputs[i]]
    
    return train_inputs, test_inputs, vocab


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

    cleaned_inputs: List[str] = list(
        map(lambda s: remove_stop_words(stem(remove_special_chars(remove_html(s))).lower()), raw_inputs))
        
    raw_labels = dataset[labels_header].to_list()

    # we will split the dataset equally between training and testing
    split_index = len(cleaned_inputs) // 2
    training_inputs = cleaned_inputs[0:split_index + 1]
    training_labels = raw_labels[0:split_index + 1]
    testing_inputs = cleaned_inputs[split_index:]
    testing_labels = raw_labels[split_index:]
    print(training_inputs[:2])
    return training_inputs, training_labels, testing_inputs, testing_labels
