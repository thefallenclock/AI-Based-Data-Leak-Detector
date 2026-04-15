# 🔐 AI-Based Data Leak Detection System

## 📌 Overview

This project is an AI-powered system that detects sensitive or spam-like text using machine learning and natural language processing.

It classifies user input into:

* **Sensitive Data 🚨**
* **Safe Text ✅**

---

## 🚀 Features

* Text preprocessing and cleaning
* TF-IDF feature extraction
* Handling imbalanced data using SMOTE
* Machine learning classification
* REST API built using FastAPI
* Real-time prediction system

---

## 🛠️ Tech Stack

* Python
* Scikit-learn
* FastAPI
* Joblib
* Pandas
* NLP (TF-IDF)

---

## ⚙️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/thefallenclock/AI-Based-Data-Leak-Detector.git
cd AI-Based-Data-Leak-Detector
```

### 2. Create virtual environment

```bash
python -m venv venv
source venv/Scripts/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the API

```bash
uvicorn app:app --reload
```

### 5. Open in browser

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Example

| Input         | Output            |
| ------------- | ----------------- |
| win money now | Sensitive Data 🚨 |
| hello bro     | Safe Text ✅       |

---

## 📈 Future Improvements

* Detect passwords, OTP, and financial data
* Improve dataset for real DLP use cases
* Add frontend UI
* Deploy online

---

## 👨‍💻 Author

Rishi
