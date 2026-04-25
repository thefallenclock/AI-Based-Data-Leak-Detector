import pandas as pd
import re
from sklearn.model_selection import train_test_split # A function to split the dataset into training and testing sets.
from sklearn.linear_model import LogisticRegression # A machine learning model for classification tasks.
# TF-IDF: Your model cannot understand text and it only understands numbers.
from sklearn.feature_extraction.text import TfidfVectorizer
# It’s a tool that: Takes text, Converts it into numbers, Based on word importance
from sklearn.metrics import accuracy_score # A function that compares real answers with model answers.
from sklearn.metrics import classification_report # a detailed report card of your model.
# SMOTE: A technique to balance dataset artificially.
# from imblearn.over_sampling import SMOTE
# Tool to save Python objects to file
import joblib


# Load the dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")
# by default, pandas uses UTF-8 encoding,
# but the dataset may contain different encoding(latin-1)
# which can cause issues when loading the data.

# Display the first 5 rows of the dataset
# print(df.head())


# print(df.columns)
# o/p: Index(['v1', 'v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], dtype='str')
# only the first 2 columns are real data, the rest are emply.

# Removing the extra columns.
df = df[["v1", "v2"]] # Keep only these two columns, ignore the rest.
# Renaming the existing columns.
df.columns = ["label", "text"] # Rename the columns to 'label' and 'text'.

# print(df.columns)
# o/p: Index(['label', 'text'], dtype='str')

# Converting the ham and spam into 0 and 1.
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# print(df['label'].unique())
# o/p: [0, 1]

# Create cleaning function to remove special characters and number from the text.
def clean_text(text):
    text = text.lower() # Conver text to lowercase.

    # DOMAIN KEYWORDS (VERY IMPORTANT)
    sensitive_keywords = ["password", "otp", "bank", "account", "pin", "ssn"]
    for word in sensitive_keywords:
        if word in text:
            text += " sensitive"

    text = re.sub(r'[^a-zA-Z]', ' ', text) # Remove special characters and numbers, keep only letters.
    return text

# Apply the cleaning function to the 'text' column.
df['text'] = df['text'].apply(clean_text)

# This is called "text preprocessing(NLP)" or "text cleaning",
# which is an important step in preparing the data for machine learning models.

# print(df['text'][0]) 
# Print the cleaned text of the first row.

# Create an instance of TfidfVectorizer
# vectorizer = TfidfVectorizer() # Think of this as: “Machine that converts text → numbers”.
# Replacing the older code with this new code:
vectorizer = TfidfVectorizer(
    stop_words = 'english', # removes useless words that add noise.
    max_features = 5000, # limits word importance to 5000.
    ngram_range=(1,2) # now it can identify pair of words too. e.g. 2gram = password is, your password.
)

x = vectorizer.fit_transform(df['text']) # Convert the 'text' column into a matrix of TF-IDF features.
y = df['label'] # The target variable (labels) for our model.

# print(x.shape)
# The shape of the feature matrix (number of messages, number of unique words).
# (5572, 8674) means there are 5572 messages and 8674 unique words in the dataset after cleaning.

# print(x)
# mathematical representation of the TF-IDF features, which is a sparse matrix.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
# Split the dataset into training and testing sets.
# 80% of the data will be used for training, and 20% will be used for testing.

# After train-test split, BEFORE training.
# smote = SMOTE() # CREATE OBJECT
# smote.fit_resample() it does 2 things:
# fit: Studies where spam points are and how they are distributed.
# resample: Creates new synthetic spam points and balances dataset
# x_train_res, y_train_res = smote.fit_resample(x_train, y_train)

# Create Model:
# model = LogisticRegression() # Create an instance of the Logistic Regression model.
# Replacing old code with the new one:
model = LogisticRegression(class_weight = 'balanced') # This parameter controls: “How important each class is”.
# This creates an empty brain (model) that we will train with our data.

# Train Model:
# Old code:
model.fit(x_train, y_train) # Train the model using the training data.
# New code:
# model.fit(x_train_res, y_train_res)
# This is where the model learns to associate the features (TF-IDF values) with the labels (spam or ham).

# “dump” means: Save something into a file
# saving the model so we don't have to train data everytime, do it once and that's done.
joblib.dump(model, 'model.pkl')

# saving the vectorizer file in the same way.
joblib.dump(vectorizer, 'vectorizer.pkl')

# Predicting answers with the trained model.
y_pred = model.predict(x_test)

# Checking Accuracy.
# print("Accuracy: ", accuracy_score(y_test, y_pred)) # o/p: Accuracy:  0.9641255605381166.

# Testing user's own sample input.
# sample = ["your password is 1234"]
# sample_vec = vectorizer.transform(sample) # Converts text into 0 and 1.
# checking the prediction.
# print(model.predict(sample_vec)) # o/p: [0].

# Even if accuracy is high (like 95%) model can still be wrong, this is called Data Imbalance.

# checking the data imbalances.
# print(df['label'].value_counts())
# o/p: label
# o/p: 0    4825
# o/p: 1     747
# o/p: Name: count, dtype: int64
# mostly data is ham(0) and very little is spam(1).

# Testing real sentences:
# samples = [
#     "hello bro how are you",
#     "win $5000 now",
#     "your password is 1234",
#     "meeting at 5pm"
# ]
# for i in samples:
#     vec = vectorizer.transform([i])
#     print(i, " -> ", model.predict(vec)[0])
# o/p: hello bro how are you  ->  0
# o/p: win $5000 now  ->  1
# o/p: your password is 1234  ->  0
# o/p: meeting at 5pm  ->  0
# We can easily say that this model is not able to detect sensitive texts, so, we have to use another model.

# print(classification_report(y_test, y_pred))
# o/p:               precision    recall  f1-score   support
# o/p: 
# o/p:            0       0.98      0.99      0.99       965
# o/p:            1       0.93      0.90      0.92       150
# o/p: 
# o/p:     accuracy                           0.98      1115
# o/p:    macro avg       0.96      0.94      0.95      1115
# o/p: weighted avg       0.98      0.98      0.98      1115
# recall is 0.90(90%) means it is still not able to detect 0.10(10%) of the data.
# since the dataset is biased and ham is more than spam, that's why we have to do spam ~ ham.

# print(classification_report(y_test, y_pred))
# o/p:               precision    recall  f1-score   support
# o/p: 
# o/p:            0       0.98      1.00      0.99       968
# o/p:            1       0.97      0.90      0.93       147
# o/p: 
# o/p:     accuracy                           0.98      1115
# o/p:    macro avg       0.98      0.95      0.96      1115
# o/p: weighted avg       0.98      0.98      0.98      1115

# print(classification_report(y_test, y_pred))
# o/p:               precision    recall  f1-score   support
# o/p: 
# o/p:            0       0.98      0.99      0.99       965
# o/p:            1       0.93      0.90      0.92       150
# o/p: 
# o/p:     accuracy                           0.98      1115
# o/p:    macro avg       0.96      0.94      0.95      1115
# o/p: weighted avg       0.98      0.98      0.98      1115

# print(classification_report(y_test, y_pred))
# o/p:               precision    recall  f1-score   support
# o/p: 
# o/p:            0       0.98      0.99      0.99       965
# o/p:            1       0.93      0.90      0.92       150
# o/p: 
# o/p:     accuracy                           0.98      1115
# o/p:    macro avg       0.96      0.94      0.95      1115
# o/p: weighted avg       0.98      0.98      0.98      1115