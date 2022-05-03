# dl-fp

### Disclaimer

You must install pandas, nltk, and beautifulsoup (`pip3 install pandas`, `pip3 install nltk`, and `pip3 install beautifulsoup`) in order to run our project.

## Introduction

[This paper](https://cs224d.stanford.edu/reports/PouransariHadi.pdf) 


## Related Work
Ah

## Data

We


### Methodology
The 

### Metrics
We plan to separate out the dataset that we found into a few sec

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