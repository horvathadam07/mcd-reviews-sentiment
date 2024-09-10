# Sentiment analysis project of McDonald's reviews

## Sources
**Python version:** 3.11.9<br/>
**Imported packages:** pandas, numpy, matplotlib, seaborn, nltk, sklearn, lightgbm<br/>
**Dataset:** https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews

## Data cleaning
After loading the data I checked duplicated observations and the missing (or incorrect) values for each feature.<br/>
The dataset was ready for EDA without any modifications.

## Feature engineering
* New feature *store_city* created from existing one *store_address*
* Features like *review_time_in_years* and *star_class* as target were transformed from the original ones
* Natural language processing techniques were applied to clean review texts

## Exploratory Data Analysis
I prepared barplots for categorical features and show insights about the sentiment of reviews in terms of stores.

![Alt text](https://github.com/horvathadam07/mcd-reviews-sentiment/blob/main/img/top10cities.png "Top 10 Cities")

Word clouds based on sentiment can be seen below:

![Alt text](https://github.com/horvathadam07/mcd-reviews-sentiment/blob/main/img/wordcloud_neg.png "Negative Word Cloud")

![Alt text](https://github.com/horvathadam07/mcd-reviews-sentiment/blob/main/img/wordcloud_pos.png "Positive Word Cloud")

## Model Building
At first I splitted the data into train and test samples with a 25% test part.<br/>
I fitted four different models and evaluated them by Accuracy score due to the balanced dataset.

To compare the model performances I also checked the polarity score by Vader's SentimentIntensityAnalyser.<br/>
Since the goal was to predict the sentiment of the text reviews that was the only predictor in the models and the Term Frequency Inverse Document Frequency transformation was used in each pipeline.

The same cross validation (k=5) was used for all models to find the optimal probability threshold.<br/>
The goal was to maximize the Accuracy score value in each iteration then the threshold was chosen as the average of the 5 train samples.

## Model Performances
The Support Vector Machine algorithm performed significantly better than the others in the test sample, the accuracy scores can be seen below:

  * **Logistic regression:** 0.8802<br/>
  * **Naive Bayes:** 0.8723<br/>
  * **Support Vector Machine:** 0.8917<br/>
  * **LightGBM:** 0.8702
