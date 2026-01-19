# Email Priority Classification System

A machine learning–based web application that classifies emails into **High**, **Medium**, or **Low** priority based on their subject and body content.

Unlike traditional spam detection, this project focuses on identifying **email importance**, similar to how real inboxes prioritize messages.

---

## 🚀 Features
- Multiclass classification: High / Medium / Low priority
- Uses Natural Language Processing (NLP)
- Deployed as a Flask web application
- Simple and clean user interface
- Model trained once and reused for predictions

---

## 🧠 How It Works
1. User enters email subject and body
2. Text is preprocessed and converted into numerical features using **TF-IDF**
3. A **Multinomial Naive Bayes** classifier predicts the priority
4. Result is displayed instantly on the web interface

---

## 🛠 Tech Stack
- **Python**
- **Scikit-learn**
- **Flask**
- **Pandas**
- **Joblib**
- **HTML & CSS**
- **Gunicorn** (for deployment)

---

## 📊 Machine Learning Details
- Feature Extraction: TF-IDF Vectorization
- Model: Multinomial Naive Bayes
- Problem Type: Multiclass Text Classification
- Evaluation: Precision, Recall, F1-score

---

