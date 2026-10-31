"""
Train the ClearSpace AI text classifier.

Expected CSV: twitter_data.csv
Required columns:
    tweet : the text of the tweet
    class : 0 = hate speech, 1 = offensive language, 2 = neither

The script saves hate_speech_model.pkl for app.py.
"""
import re
import string
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = "twitter_data.csv"
MODEL_PATH = "hate_speech_model.pkl"

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df = pd.read_csv(DATA_PATH)

if "tweet" not in df.columns or "class" not in df.columns:
    raise ValueError("CSV must contain 'tweet' and 'class' columns.")

df = df[["tweet", "class"]].dropna()
df["class"] = pd.to_numeric(df["class"], errors="coerce")
df = df.dropna(subset=["class"])
df["class"] = df["class"].astype(int)
df = df[df["class"].isin([0, 1, 2])]
df["tweet"] = df["tweet"].apply(clean_text)
df = df[df["tweet"].str.len() > 0]

if df["class"].nunique() < 2:
    raise ValueError("The dataset needs at least two different classes.")

X = df["tweet"]
y = df["class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", DecisionTreeClassifier(random_state=42, class_weight="balanced"))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nClassification report:")
print(classification_report(
    y_test, predictions,
    labels=[0, 1, 2],
    target_names=["Hate speech", "Offensive language", "Neither"],
    zero_division=0
))
print("Confusion matrix (rows = actual, columns = predicted):")
print(confusion_matrix(y_test, predictions, labels=[0, 1, 2]))

joblib.dump(model, MODEL_PATH)
print(f"\nSaved trained model as {MODEL_PATH}")
