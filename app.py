"""Flask application for the tweet emotion and sentiment dashboard."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import joblib
from flask import Flask, jsonify, render_template, request, send_from_directory


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "tweet_sentiment_tfidf_svm.joblib"
METRICS_PATH = PROJECT_DIR / "outputs" / "metrics.json"
TERMS_PATH = PROJECT_DIR / "outputs" / "top_weighted_terms.csv"

app = Flask(__name__)

# These assets are committed with the application so a fresh deployment works
# immediately, without retraining the model on the server.
model = joblib.load(MODEL_PATH)
with METRICS_PATH.open(encoding="utf-8") as metrics_file:
    metrics = json.load(metrics_file)
with TERMS_PATH.open(newline="", encoding="utf-8") as terms_file:
    top_terms = [
        {**row, "weight": float(row["weight"])}
        for row in csv.DictReader(terms_file)
    ]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_TEXT_RE = re.compile(r"[^a-z\s']")
WHITESPACE_RE = re.compile(r"\s+")


def clean_tweet(text: str) -> str:
    """Normalize tweet-specific noise in the same way as the training flow."""
    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = text.replace("&amp;", "and").replace("n't", " not")
    text = NON_TEXT_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


@app.get("/")
def index():
    """Render the dashboard."""
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    """Serve the root favicon."""
    return send_from_directory(PROJECT_DIR / "static", "favicon.ico", mimetype="image/vnd.microsoft.icon")


@app.get("/manifest.webmanifest")
def manifest():
    """Serve the web app manifest for Chrome home screen/bookmark support."""
    return send_from_directory(PROJECT_DIR / "static", "manifest.webmanifest", mimetype="application/manifest+json")


@app.get("/apple-touch-icon.png")
def apple_touch_icon():
    """Serve the touch icon for mobile and desktop shortcuts."""
    return send_from_directory(PROJECT_DIR / "static", "apple-touch-icon.png", mimetype="image/png")


@app.get("/api/metrics")
def get_metrics():
    """Return the recorded held-out evaluation metrics."""
    report = metrics["classification_report"]
    return jsonify(
        {
            "total_tweets": metrics["total_tweets"],
            "training_tweets": metrics["training_tweets"],
            "test_tweets": metrics["test_tweets"],
            "accuracy": metrics["test_accuracy"],
            "negative_precision": round(report["Negative"]["precision"], 4),
            "positive_precision": round(report["Positive"]["precision"], 4),
            "negative_recall": round(report["Negative"]["recall"], 4),
            "positive_recall": round(report["Positive"]["recall"], 4),
        }
    )


@app.get("/api/top-terms")
def get_top_terms():
    """Return the strongest positive and negative model indicators."""
    return jsonify(
        {
            "negative": [term for term in top_terms if term["sentiment"] == "Negative"][:10],
            "positive": [term for term in top_terms if term["sentiment"] == "Positive"][:10],
        }
    )


@app.post("/api/predict")
def predict():
    """Classify a short piece of tweet-like text."""
    data = request.get_json(silent=True) or {}
    text = str(data.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Please enter some text to analyse."}), 400
    if len(text) > 280:
        return jsonify({"error": "Please keep the text to 280 characters or fewer."}), 400

    cleaned = clean_tweet(text)
    if not cleaned:
        return jsonify({"error": "Enter text containing at least one word."}), 400

    score = float(model.decision_function([cleaned])[0])
    sentiment = str(model.predict([cleaned])[0])
    confidence = abs(score) / (1 + abs(score))

    return jsonify(
        {
            "text": text,
            "sentiment": sentiment,
            "score": round(score, 3),
            "confidence": round(confidence, 3),
            "emoji": "😊" if sentiment == "Positive" else "😞",
        }
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
