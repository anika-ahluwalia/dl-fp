# dl-fp

### Disclaimer

You must install pandas, nltk, and beautifulsoup (`pip3 install pandas`, `pip3 install nltk`, and `pip3 install beautifulsoup`) in order to run our project.

## Introduction

Sentiment analysis applies natural language processing (NLP) to determine the affect of text documents. It is useful for determining how well emotion is conveyed via text, which is particularly applicable for authors who want readers to empathize with their experiences (eg. reviewers). Because our group is interested in NLP and enjoys watching films, we decided to base our final project on a paper that studies sentiment analysis of movie reviews.
For our final project, we use binary classification to map movie reviews to a positive or negative sentiment, using a dataset of 50,000 IMDB movie reviews for training and testing. We compare two types of models – bag of words and word2vec models – using different classifiers (random forest vs SVM) and feature mappings (averaging vs clustering) to compare their accuracies on the same dataset.

## Related Work

Our project was based on [this Stanford paper](https://cs224d.stanford.edu/reports/PouransariHadi.pdf). The study compares different NLP models with various classifiers to examine the accuracy of sentiment analysis on movie reviews. Results showed that the word2vec model with averaging feature mapping and a logistic regression classifier was the most accurate (86.6%), although most models hovered at around 84-85% accuracy. Although sentiment analysis in any context poses an interesting problem, we decided to choose this paper because it is particularly interesting in the context of movie reviews, since horror/rom-com movies are usually described using inherently positive/negative words (eg. "scary," "gory," "love," "happy").

One related paper is [this article on sentiment analysis of tweets](https://uksim.info/icaiet2014/CD/data/7910a212.pdf). It presents an interesting issue because tweets are limited to 140 characters, which may result in different results from sentiment analysis of regular text. Additionally, tweets have different sentence structures from academic texts and websites because of commonly-used slang and acronyms, which means training datasets must also include slang/acronym.

Here is a running list of publicly available implementations:
[Sentiment Analysis of Movie Reviews](https://www.kaggle.com/code/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews/notebook)
[Movie Review Sentiment Analysis EDA and Models](https://www.kaggle.com/code/artgor/movie-review-sentiment-analysis-eda-and-models#Deep-learning)

## Data

We will be using the IMDB Dataset of 50K Movie Reviews to train our binary classification model, which labels reviews as either positive or negative sentiment. As described in the paper, we will clean up the data during preprocessing to remove stop words, tags, irrelevant punctuation, and convert words to lowercase as needed.

## Methodology

### Model

Because we are implementing binary classification, we will implement two different models with two different feature vector mappings to compare the models’ accuracies on the same IMDB dataset. 
First, we will use the preprocessed data to generate feature vectors using both bag of words and word2vec. Then, for word2vec, we will map the feature vectors to movie reviews by either (1) averaging the word vectors of all words in a review, or (2) clustering similar words together to find the differences between clusters.
Once we’ve generated all feature vectors and mapped them to movie reviews, we will implement different classifiers – namely random forest and SVM – to compare different combinations of models, mappings, and classifiers.

### Training/Testing

The model will be trained on the IMDB Dataset with 25,000 training data points, and we will test the model on the IMDB Dataset with 25,000 test data points.
We plan to test the following four model-classification combinations:
Bag of words + random forest
Bag of words + SVM
Word2vec + averaging + random forest
Word2vec + clustering + random forest
After training and testing the four combinations, we will compute the accuracy of each test set and compare the accuracies across models.

### Metrics

We plan to separate out the dataset that we found into a few sec

## Goals
**Base:** We would like to create a bag of words model that can successfully analyze the sentiment in movie reviews on IMDB.

**Target:** We hope to expand to a bag of words and word-2-vec model that is able to analyze the sentiment of movie reviews from IMDB with at least 75% accuracy.

**Stretch:** We hope to be able to use our model's sentiment analysis to predict the number of stars that a particular movie review on IMDB received.

## Ethics

### What broader societal issues are relevant to your chosen problem space?
Ou

### What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?
Our dataset is a collection 


## Division of labor

Preprocessing the data – Anika + Galen

Writing a modified word-2-vec model – Galen + Naomi

Writing a bag of words model – Anika + Lauren

Training and Testing – Lauren + Naomi

Accuracy computations – Anika

