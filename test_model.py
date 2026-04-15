# joblib is a tool to save files so that we do not have to train the model again and again.
# just train once and then load everytime instantly.
import joblib
# re is the regular expression library, which is used for text cleaning.
import re

# LOAD SAVED FILES:
# load saved model and vectorizer, which is already trained once.
loaded_model = joblib.load("model.pkl")
loaded_vectorizer = joblib.load("vectorizer.pkl")

# TEXT CLEANING FUNCTION:
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

# TEST INPUT:
# sample input.
sample = ["hello bro"]

# Clean input(Important)
cleaned = [clean_text(sample[0])]

# Covert text into numbers.
vec = loaded_vectorizer.transform(cleaned)

# Predict
result = loaded_model.predict(vec)

# printing the result.
print(result)