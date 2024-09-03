# Sentiment Analysis Project
This repository contains code for performing sentiment analysis on a text dataset using machine learning models. The project involves data preprocessing, feature extraction, model training, evaluation, and prediction on new data.

## Key Components of the Code
### 1-Data Loading and Initial Exploration:
The dataset is loaded from a CSV file using pandas.
An exploratory data analysis (EDA) report is generated using ydata_profiling to get an overview of the data.
Missing values and data types are examined, and basic statistical summaries are provided.

### 2-Data Visualization:
Seaborn and Matplotlib are used to visualize the pairwise relationships between numerical features.
A bar chart is plotted to show the distribution of samples in each class (target variable).

### 3-Data Preprocessing:
Text preprocessing is performed using the spaCy library to lemmatize the text and remove stop words.
Unnecessary columns (ID and Source) are dropped from the dataset.

### 4-Feature Extraction:
The CountVectorizer from sklearn is used to convert the processed text data into numerical features suitable for machine learning models.

### 5-Model Training and Evaluation:
The dataset is split into training and testing sets using train_test_split.
Two models are trained:
Support Vector Classifier (SVC): Hyperparameter tuning is performed using GridSearchCV to find the best model.
Artificial Neural Network (MLPClassifier): Hyperparameter tuning is also applied using GridSearchCV.
Both models are evaluated using the classification_report to show precision, recall, and F1-score.

### 6-Model Saving and Loading:

The best-performing SVC model is saved to a file using joblib.
The model is loaded back from the file to make predictions on new, unlabeled text data.

### 7-Prediction on New Data:

New text samples are processed and vectorized.
The loaded model is used to predict the sentiment of these new text samples, demonstrating the model's application to real-world data.

## How to Use
1-Clone the repository and navigate to the project directory.
2-Ensure that all required Python packages are installed (use the requirements.txt file if available).
3-Run the script to load the dataset, preprocess the data, train the models, and generate predictions.

## Requirements
Python 3.x
pandas
spaCy
Matplotlib
Seaborn
tabulate
scikit-learn
ydata-profiling
joblib

## Author
Ahmed Ammar

Feel free to contribute or raise issues if you encounter any problems!
