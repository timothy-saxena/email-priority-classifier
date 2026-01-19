import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

# 1. Load dataset
data = pd.read_csv("data/emails.csv")

# 2. Combine subject and body
X = data["subject"].astype(str) + " " + data["body"].astype(str)
y = data["label"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create ML pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("classifier", MultinomialNB())
])

# 5. Train model
model.fit(X_train, y_train)

# 6. Evaluate model
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# 7. Save trained model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/email_priority_model.pkl")

print("\nModel saved successfully in model/email_priority_model.pkl")
