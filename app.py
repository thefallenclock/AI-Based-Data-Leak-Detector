from fastapi import FastAPI # FastAPI is the library to create API.
import joblib # a tool to save and load trained model.
import re # regular expression.

app = FastAPI() # creates your server.

# Loads your trained model so we do not need to train again.
loaded_model = joblib.load("model.pkl") # AI brain of our model.

# Converts text to numbers that must match training.
loaded_vectorizer = joblib.load("vectorizer.pkl") # This is your translator.

# Same as cleaning as training. It ensures consistency.
# Without this predictions would be wrong.
def clean_text(text):
    text = text.lower() # converts text into lowercase
    text = re.sub(r'[^a-zA-Z]', ' ', text) # replace all the non-alphabetic characters to ' ' from text.
    return text

# CREATE ENDPOINT: An endpoint is a URL where your function runs.
@app.get("/predict") # When someone visits /predict, run the function below.
def predict(text: str):
    cleaned = [clean_text(text)]
    vec = loaded_vectorizer.transform(cleaned)
    result = loaded_model.predict(vec)[0]
    if result == 1:
        return {"prediction": "Sensitive Data Detected!!!"}
    else:
        return {"prediction": "Safe Text!!!"}