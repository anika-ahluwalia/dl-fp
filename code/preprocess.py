import csv
import os
from typing import Tuple, List, Dict

import gensim
import nltk.stem
from bs4 import BeautifulSoup
import pandas as pd
import string
import re
import numpy as np
from tqdm import tqdm
import tensorflow_cloud as tfc


def words_to_ids(review_list: List[List[str]], vocab_dict: Dict[str, int]) -> List[List[int]]:
    words_as_ids = []
    for review in review_list:
        review_with_ids = []
        for word in review:
            review_with_ids.append(vocab_dict[word])
        words_as_ids.append(review_with_ids)

    return words_as_ids


def word2vec_preprocess(review_list: List[List[str]], vocab_dict: Dict[str, int], window_size: int) -> List[List[int]]:
    """
    Takes a list of reviews, which are each a list of words, and generates a list of skipgrams.

    :param review_list: A list of reviews, each of which is a list of strings (words).
    :param vocab_dict: A dictionary mapping words to their numeric ID.
    :param window_size: For each word in a review, how many other words in the review to consider for skipgrams.
    :return: A list of skipgrams generated from the corpus.
    """
    skipgrams: List[List[int]] = []
    for review in tqdm(review_list):
        for word_index, word in enumerate(review):
            min_idx = max(0, word_index - window_size)
            max_idx = min(len(review), word_index + window_size + 1)
            for nb_word in review[min_idx:max_idx]:
                if nb_word != word:
                    skipgrams.append([vocab_dict[word], vocab_dict[nb_word]])
    return skipgrams


def word2vec_sentiment_preprocess(review_list: List[List[int]], word2vec_model) -> np.ndarray:
    review_embeddings = []
    for review in review_list:
        word_embeddings = []
        for word in review:
            word_embeddings.append(word2vec_model.wv[word])
        review_embeddings.append(np.average(np.array(word_embeddings), axis=0))

    return np.array(review_embeddings)


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


def build_vocab(inputs: List[List[str]]) -> Dict[str, int]:
    """
    Creates a dictionary mapping all unique words in the corpus to a numeric ID, 0 through (# of unique words - 1).

    :param inputs: A list of reviews, each of which is a list of words.
    :return: A dictionary mapping words to their numeric ID.
    """
    vocab = {}

    all_words = []

    for review in inputs:
        for word in review:
            all_words.append(word)

    unique_inputs = np.unique(all_words)

    # convert unique input words to unique IDs
    for i in range(len(unique_inputs)):
        vocab[unique_inputs[i]] = i

    print(vocab)

    return vocab


def get_data(file_path: str, cleaned_file_path: str, inputs_header: str, labels_header: str) -> Tuple[
    List[List[str]], List[int], List[List[str]], List[int], Dict[str, int]]:
    """
    Handles processing corpus of IMDb reviews and sentiment labels into training and testing datasets.

    :param file_path: The path to the CSV file containing the IMDb dataset.
    :param cleaned_file_path: The path to the CSV file containing the cleaned IMDb dataset.
    :param inputs_header: The name of the column containing IMDb reviews.
    :param labels_header: The name of the column containing reviews' sentiment labels.
    :return: Four lists: training inputs, training labels, testing inputs, and testing labels.
    """
    dataset: pd.DataFrame = pd.read_csv(file_path)
    raw_inputs: List[str] = dataset[inputs_header].to_list()
    raw_labels = dataset[labels_header].to_list()

    cleaned_inputs: List[str]
    if not os.path.exists(cleaned_file_path) and not tfc.remote():
        # encode the labels as positive or negative
        cleaned_labels = []
        for label in raw_labels:
            if label == 'positive':
                cleaned_labels.append(1)
            elif label == 'negative':
                cleaned_labels.append(0)

        cleaned_inputs = list(
            map(lambda s: remove_stop_words(stem(remove_special_chars(remove_html(s))).lower()), raw_inputs))

        with open(cleaned_file_path, "w") as cleaned_csv:
            csv_writer = csv.writer(cleaned_csv)
            csv_writer.writerow(["cleaned_review", "sentiment"])
            csv_writer.writerows(list(zip(cleaned_inputs, cleaned_labels)))
    else:
        cleaned_dataset = pd.read_csv(cleaned_file_path)
        cleaned_inputs = cleaned_dataset["cleaned_review"].to_list()
        cleaned_labels = cleaned_dataset["sentiment"].to_list()

    ready_inputs = []
    for review in cleaned_inputs:
        review_as_list = review.split()
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
