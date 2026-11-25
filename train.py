"""Train and evaluate a spam classifier using actual email content."""
import html
import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_FILE = Path(__file__).with_name("enron_spam_data.zip")
LABELS = ["ham", "spam"]


def clean_email(text):
    """Lowercase and remove HTML tags, addresses, links and punctuation."""
    text = html.unescape(text).lower()
    text = re.sub(r"</?[a-z][^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+|\S+@\S+", " ", text)
    # Use the same word boundaries here and in the vectorizer.
    return " ".join(re.findall(r"\b\w\w+\b", text))


def load_emails():
    # Pandas reads the single CSV inside this ZIP without extracting it.
    emails = pd.read_csv(DATA_FILE, usecols=["Subject", "Message", "Spam/Ham"])
    emails = emails[emails["Spam/Ham"].isin(LABELS)].copy()
    combined = emails["Subject"].fillna("") + " " + emails["Message"].fillna("")
    emails["text"] = combined.map(clean_email)
    emails = emails[emails["text"] != ""]
    # Identical cleaned emails must not appear in both training and testing.
    label_count = emails.groupby("text")["Spam/Ham"].transform("nunique")
    return emails[label_count == 1].drop_duplicates("text")


def train_model():
    emails = load_emails()
    x_train, x_test, y_train, y_test = train_test_split(
        emails["text"], emails["Spam/Ham"],
        test_size=0.2, random_state=42, stratify=emails["Spam/Ham"]
    )
    # Fit the vocabulary and classifier using only training emails.
    model = Pipeline([
        ("tfidf", TfidfVectorizer(min_df=2, max_features=20000)),
        ("classifier", MultinomialNB(alpha=1.0)),
    ])
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    results = {
        "accuracy": accuracy_score(y_test, predictions),
        "correct": int((predictions == y_test).sum()),
        "train_size": len(x_train),
        "test_size": len(x_test),
        "counts": emails["Spam/Ham"].value_counts().reindex(LABELS),
        "report": classification_report(y_test, predictions, labels=LABELS, digits=3),
        "matrix": confusion_matrix(y_test, predictions, labels=LABELS),
    }
    return model, results


if __name__ == "__main__":
    if not DATA_FILE.is_file():
        raise SystemExit(f"Missing {DATA_FILE.name}. See step 5 of README.md.")
    _, results = train_model()
    print(results["counts"].to_string())
    print(f"Training emails: {results['train_size']}")
    print(f"Test accuracy: {results['correct']}/{results['test_size']} "
          f"({results['accuracy']:.2%})")
    print(results["report"])
    print("Confusion matrix (rows = actual, columns = predicted; order: ham, spam)")
    print(results["matrix"])
