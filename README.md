# TWITTER-SENTIMENT-ANALYSIS-DURING-COVID-LOCKDOWN
I've created a Python application that allows users to input a Tweet, and it leverages an open-source sentiment analysis dataset to identify the sentiment associated with that Tweet. This application combines natural language processing techniques with readily available sentiment data to determine whether the Tweet is generally associated with positive, negative, or neutral sentiments. Users can gain valuable insights into public sentiment about a specific topic or keyword, making it a useful tool for monitoring online sentiment trends, social media analysis, and market research. Additionally, the application can be further enhanced by integrating with real-time data sources or more extensive sentiment analysis models for improved accuracy and insights.

STEPS INVOLVED ARE :

1. IMPORTING LIBRARIES AND DATASETS:
Begin by importing the necessary Python libraries such as pandas, numpy, nltk, scikit-learn, and matplotlib.
Load the sentiment analysis dataset(s) from open-source repositories into dataframes.

2. EXPLORING DATASETS:
Conduct exploratory data analysis (EDA) to understand the structure of the data.
Check for data shape, column names, and the distribution of sentiment labels.

3. REMOVING UNWANTED FEATURES:
Identify and remove any irrelevant or redundant features from the dataset that won't contribute to sentiment analysis.

4. COMBINE THE 3 DATAFRAMES:
If you have multiple datasets, merge them into a single dataframe for analysis, ensuring that the schema matches.

5. DATA PREPROCESSING - CLEANING OF TWEETS:
Perform data cleaning to remove special characters, URLs, and any noise from the text data.
Convert text to lowercase to ensure consistency.
Remove stop words (common words like "the," "and," etc.) as they don't carry much sentiment information.

6. STEMMING USING PORTER STEMMER:
Apply stemming using the Porter Stemmer algorithm to reduce words to their root forms. This helps in simplifying the text data.

7. VISUALIZATION OF DATA:
Create visualizations such as bar graphs and pie charts to visualize the distribution of sentiment labels in the dataset.
Analyze the data's class imbalance, if any.

8. VECTORIZATION USING TF-IDF:
Convert the preprocessed text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization.

9. TRAIN-TEST SPLIT:
Split the dataset into training and testing sets to evaluate the model's performance. Typically, a 70-30 or 80-20 split is used.

10. LOGISTIC REGRESSION CLASSIFIER AND APPLYING GRID SEARCH:
Implement a Logistic Regression classifier and optimize its hyperparameters using grid search.
Train the model on the training data and evaluate its performance on the test data.

11. SUPPORT VECTOR CLASSIFIER AND APPLYING GRID SEARCH:
Implement a Support Vector Classifier (SVC) and optimize its hyperparameters using grid search.
Train the model on the training data and evaluate its performance on the test data.

12. MULTINOMIAL NAIVE BAYES AND APPLYING GRID SEARCH:
Implement a Multinomial Naive Bayes classifier and optimize its hyperparameters using grid search.
Train the model on the training data and evaluate its performance on the test data.

13. DECISION TREE CLASSIFIER:
Implement a Decision Tree classifier without hyperparameter tuning.
Train the model on the training data and evaluate its performance on the test data.

14. ACCURACY OF THE MODELS:
Record and present the accuracy, precision, recall, F1-score, and any other relevant metrics for each model.
Display these metrics in a tabular form for easy comparison.
Create bar graphs to visually represent the performance of different models
