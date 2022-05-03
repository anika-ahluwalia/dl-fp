# Analyzing Sentiment in IMDB Movie Reviews

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

We plan to separate out the dataset that we found into two sections: one for training and one for testing. The dataset comes with reviews categorized into one of two sentiments; positive or negative. We will begin by training our model on half of the data and then seeing how the sentiment matches up. Because we are solving a problem of binary classification, our accuracy measure will be fairly simple. We can simply look at the percentage of examples in our training data that are matched to the correct label. The authors of the paper also used this accuracy measure when computing the accuracy of their model.

As we expand our project into our reach goal (predicting the number of stars that a review received), the accuracy measure will get a bit more complicated. We will take a measure of how close the rating is (exactly the correct number of stars, one star off, two stars off, etc) and will use that to compute how accurate our model is. 

## Goals
**Base:** We would like to create a bag of words model that can successfully analyze the sentiment in movie reviews on IMDB.

**Target:** We hope to expand to a bag of words and word-2-vec model that is able to analyze the sentiment of movie reviews from IMDB with at least 75% accuracy.

**Stretch:** We hope to be able to use our model's sentiment analysis to predict the number of stars that a particular movie review on IMDB received.

## Ethics

### Why is Deep Learning a good approach to this problem?
Deep Learning is a good approach to this problem because deep learning is particularly good at processing language and learning about sequential patterns in related information. For example, in class we learned about deep learning's applications into Natural Language Processing (NLP). We learned that there are a variety of deep learning models that could solve this problem and create models that mimick how one would naturally speak in a certain language. This is because deep learning models naturally store information based on what they have seen and use that to make predictions. Thus, by naturally storing information about how sentiment is expressed in reviews and communcications, the model would be a really good predictor of this in future reviews. This is a good way to create an adaptable model that can be specialized on this specific dataset for movies, but could also be generalized to a wide variety of other applications if it was trained on different data. 

### What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?
Our dataset is a collection movie reviews from the IMDB website and includes the sentiment indicated in each review. The dataset could be concerning in how it was collected because it is only taken from English Hollywood movies and so our model would likely learn to analyse sentimemt with American vernacular and slang. This means that our model would not be applicable to a wide variety of countries and types of films and therefore could contain societal bias. Additionally, the dataset only provides binary classification into "positive" or "negative" sentiment. This could potentially be worrisome because in real life people generally have both positive and negative words in a sentence and could end up conveying an emotion that is neutral or different than those two. Our model would not be able to account for that and would simply have to classify the review into one of these two categories. Overall, the dataset appears to be representative for fairly straightforward reviews of Hollywood movies, but beyond that specific application, would include severe bias towards this sub-group and would be lacking in knowledge. 


## Division of labor

Preprocessing the data – Anika + Galen

Writing a modified word-2-vec model – Galen + Naomi

Writing a bag of words model – Anika + Lauren

Training and Testing – Lauren + Naomi

Accuracy computations – Anika

## Reflection
Our reflection for mentor check-in 3 can be found [here](https://docs.google.com/document/d/100yG-2A6vtRPgLJNLJhYUCCDILl56G8BkJxYx0_JI_0/edit?usp=sharing).

