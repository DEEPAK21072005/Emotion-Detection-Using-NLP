"""Train and use a reproducible TF-IDF tweet sentiment classifier.

The project uses Sentiment140's labelled Twitter data (negative = 0,
positive = 4) and deliberately excludes the neutral class because that
corpus does not provide a neutral label.  Run `python sentiment_analysis.py
train --download` for a complete training and evaluation run.
"""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # Write charts to disk; works on machines without a GUI.
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DIR = PROJECT_DIR / "data" / "raw"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"
MODEL_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "outputs"
MODEL_PATH = MODEL_DIR / "tweet_sentiment_tfidf_svm.joblib"

SENTIMENT140_URL = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
SENTIMENT140_CSV = RAW_DIR / "training.1600000.processed.noemoticon.csv"
COLUMN_NAMES = ["target", "id", "date", "query", "user", "text"]

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_TEXT_RE = re.compile(r"[^a-z\s']")
WHITESPACE_RE = re.compile(r"\s+")


def clean_tweet(text: object) -> str:
    """Normalize tweet-specific noise while retaining sentiment-bearing words."""
    text = str(text).lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = text.replace("&amp;", "and").replace("n't", " not")
    text = NON_TEXT_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def ensure_directories() -> None:
    for directory in (RAW_DIR, PROCESSED_DIR, MODEL_DIR, OUTPUT_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def download_sentiment140() -> Path:
    """Download and safely extract Sentiment140's official training archive."""
    ensure_directories()
    if SENTIMENT140_CSV.exists():
        return SENTIMENT140_CSV

    archive = RAW_DIR / "sentiment140.zip"
    print(f"Downloading Sentiment140 data from {SENTIMENT140_URL} ...")
    request = urllib.request.Request(
        SENTIMENT140_URL,
        headers={"User-Agent": "TweetSentimentPortfolioProject/1.0"},
    )
    with urllib.request.urlopen(request, timeout=180) as response, archive.open("wb") as file:
        while chunk := response.read(1024 * 1024):
            file.write(chunk)

    with zipfile.ZipFile(archive) as zipped:
        member = next(
            (name for name in zipped.namelist() if name.endswith("training.1600000.processed.noemoticon.csv")),
            None,
        )
        if member is None:
            raise RuntimeError("The downloaded archive did not contain the expected training CSV.")
        destination = SENTIMENT140_CSV.resolve()
        for info in zipped.infolist():
            if info.filename == member:
                candidate = (RAW_DIR / info.filename).resolve()
                if candidate != destination or RAW_DIR.resolve() not in candidate.parents:
                    raise RuntimeError("Unsafe archive path encountered.")
                with zipped.open(info) as source, destination.open("wb") as target:
                    target.write(source.read())
                break

    print(f"Dataset ready: {SENTIMENT140_CSV}")
    return SENTIMENT140_CSV


def build_balanced_sample(dataset_path: Path, sample_size: int, random_state: int) -> pd.DataFrame:
    """Read labels/text only and produce a balanced, cleaned, reproducible sample."""
    if sample_size < 2_000:
        raise ValueError("Use at least 2,000 tweets so the evaluation is meaningful.")
    if sample_size % 2:
        sample_size -= 1
        print(f"Sample size adjusted to {sample_size} to keep the classes balanced.")

    print("Reading the labelled tweet corpus (this may take a minute on the first run) ...")
    data = pd.read_csv(
        dataset_path,
        encoding="latin-1",
        names=COLUMN_NAMES,
        usecols=[0, 5],
        dtype={"target": "int8", "text": "string"},
    ).dropna(subset=["text"])
    data = data[data["target"].isin([0, 4])].copy()
    data["label"] = data["target"].map({0: "Negative", 4: "Positive"})

    per_class = sample_size // 2
    available = data["label"].value_counts()
    
    # If requested sample is too large, use maximum available
    if (available < per_class).any():
        actual_per_class = available.min()
        sample_size = actual_per_class * 2
        per_class = actual_per_class
        print(f"Sample size adjusted to {sample_size:,} (maximum available per class: {actual_per_class:,})")

    # A very small number of tweets contain only links, handles, or symbols and
    # become empty after cleaning.  Draw a buffer, then take the requested
    # count after that legitimate filtering step.
    # For large samples, use a larger buffer (5-10% loss rate for cleaning)
    buffer_ratio = 0.10 if sample_size > 500_000 else 0.02
    candidate_per_class = min(int(per_class * (1 + buffer_ratio)), available.min())
    
    sampled = (
        data.groupby("label", group_keys=False)
        .sample(n=candidate_per_class, random_state=random_state)
        .loc[:, ["text", "label"]]
        .rename(columns={"text": "tweet"})
    )
    sampled["clean_tweet"] = sampled["tweet"].map(clean_tweet)
    sampled = sampled[sampled["clean_tweet"].str.len() > 0].copy()
    remaining = sampled["label"].value_counts()
    if (remaining.reindex(["Negative", "Positive"], fill_value=0) < per_class).any():
        # If still not enough after cleaning, just take what we can
        actual_remaining = remaining.min()
        print(f"Warning: After cleaning, only {actual_remaining:,} tweets per class are available. Using that instead.")
        per_class = actual_remaining
        sample_size = per_class * 2
    
    sampled = (
        sampled.groupby("label", group_keys=False)
        .sample(n=per_class, random_state=random_state)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    output_file = PROCESSED_DIR / f"sentiment140_balanced_{sample_size}.csv"
    sampled.to_csv(output_file, index=False, encoding="utf-8")
    print(f"Prepared {len(sampled):,} balanced tweets: {output_file}")
    return sampled


def make_pipeline(large_dataset: bool = False) -> Pipeline:
    """Return a strong, fast baseline for short informal social-media text.
    
    Args:
        large_dataset: If True, uses optimized hyperparameters for large-scale training (>500k tweets).
    """
    if large_dataset:
        # Optimized for large datasets (1.6M tweets)
        tfidf_params = {
            "ngram_range": (1, 2),
            "min_df": 3,  # Require at least 3 occurrences (less noise from rare words)
            "max_df": 0.95,  # Remove very common words
            "max_features": 200_000,  # More features for better vocabulary coverage
            "sublinear_tf": True,
            "strip_accents": "unicode",
        }
        svm_params = {"C": 0.8, "class_weight": "balanced", "dual": False, "max_iter": 2000}
    else:
        # Original parameters for smaller datasets
        tfidf_params = {
            "ngram_range": (1, 2),
            "min_df": 2,
            "max_df": 0.98,
            "max_features": 100_000,
            "sublinear_tf": True,
            "strip_accents": "unicode",
        }
        svm_params = {"C": 1.2, "class_weight": "balanced"}
    
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(**tfidf_params)),
            ("classifier", LinearSVC(**svm_params)),
        ]
    )


def save_visualizations(y_test: pd.Series, predictions: list[str], model: Pipeline) -> None:
    """Save publication-ready evaluation and feature-importance charts."""
    sns.set_theme(style="whitegrid", context="notebook")
    labels = ["Negative", "Positive"]

    figure, axis = plt.subplots(figsize=(6, 5))
    display = ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions, labels=labels), display_labels=labels
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title("Tweet Sentiment Classifier — Confusion Matrix")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close(figure)

    vectorizer: TfidfVectorizer = model.named_steps["tfidf"]
    classifier: LinearSVC = model.named_steps["classifier"]
    terms = vectorizer.get_feature_names_out()
    weights = classifier.coef_[0]
    top_negative = sorted(zip(weights, terms))[:15]
    top_positive = sorted(zip(weights, terms), reverse=True)[:15]
    influential = pd.DataFrame(
        [
            {"term": term, "weight": weight, "sentiment": "Negative"}
            for weight, term in top_negative
        ]
        + [
            {"term": term, "weight": weight, "sentiment": "Positive"}
            for weight, term in top_positive
        ]
    ).sort_values("weight")
    influential.to_csv(OUTPUT_DIR / "top_weighted_terms.csv", index=False)

    figure, axis = plt.subplots(figsize=(10, 8))
    palette = {"Negative": "#d1495b", "Positive": "#2a9d8f"}
    sns.barplot(data=influential, x="weight", y="term", hue="sentiment", palette=palette, dodge=False, ax=axis)
    axis.set_title("Most Influential TF-IDF Terms")
    axis.set_xlabel("Linear SVM coefficient")
    axis.set_ylabel("")
    axis.legend(title="Class", loc="lower right")
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "top_weighted_terms.png", dpi=180)
    plt.close(figure)


def train(args: argparse.Namespace) -> None:
    ensure_directories()
    dataset_path = Path(args.data).expanduser().resolve() if args.data else SENTIMENT140_CSV
    if not dataset_path.exists():
        if not args.download:
            raise FileNotFoundError(
                "Dataset not found. Run with --download, or pass --data PATH_TO_SENTIMENT140_CSV."
            )
        dataset_path = download_sentiment140()

    sample = build_balanced_sample(dataset_path, args.sample_size, args.random_state)
    x_train, x_test, y_train, y_test = train_test_split(
        sample["clean_tweet"],
        sample["label"],
        test_size=args.test_size,
        stratify=sample["label"],
        random_state=args.random_state,
    )
    # Use optimized hyperparameters for large datasets
    large_dataset = args.sample_size > 500_000
    model = make_pipeline(large_dataset=large_dataset)
    print(f"Training on {len(x_train):,} tweets; evaluating on {len(x_test):,} held-out tweets ...")
    print(f"Using {'optimized' if large_dataset else 'standard'} hyperparameters ...")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, labels=["Negative", "Positive"], output_dict=True)

    joblib.dump(model, MODEL_PATH)
    test_predictions = pd.DataFrame(
        {
            "tweet": sample.loc[x_test.index, "tweet"],
            "actual": y_test,
            "predicted": predictions,
            "correct": y_test.to_numpy() == predictions,
        }
    )
    test_predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8")
    metrics = {
        "dataset": "Sentiment140 (balanced random sample)",
        "total_tweets": int(len(sample)),
        "training_tweets": int(len(x_train)),
        "test_tweets": int(len(x_test)),
        "random_state": args.random_state,
        "test_accuracy": round(float(accuracy), 4),
        "large_dataset_optimization": large_dataset,
        "classification_report": report,
    }
    (OUTPUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_visualizations(y_test, predictions, model)

    print("\nTraining complete")
    print(f"Held-out test accuracy: {accuracy:.2%}")
    print(classification_report(y_test, predictions, labels=["Negative", "Positive"]))
    print(f"Model: {MODEL_PATH}")
    print(f"Results: {OUTPUT_DIR}")


def predict(args: argparse.Namespace) -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No saved model found. Train one first with: python sentiment_analysis.py train --download")
    model: Pipeline = joblib.load(MODEL_PATH)
    cleaned = clean_tweet(args.text)
    if not cleaned:
        raise ValueError("Enter some tweet text containing words to analyze.")
    score = float(model.decision_function([cleaned])[0])
    sentiment = model.predict([cleaned])[0]
    print(f"Tweet: {args.text}")
    print(f"Sentiment: {sentiment}")
    print(f"Decision score: {score:.3f} (positive = Positive, negative = Negative)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TF-IDF + Linear SVM tweet sentiment analysis")
    commands = parser.add_subparsers(dest="command", required=True)
    trainer = commands.add_parser("train", help="prepare data, train, evaluate, and save a model")
    trainer.add_argument("--download", action="store_true", help="download Sentiment140 if it is not already present")
    trainer.add_argument("--data", help="path to a Sentiment140-format training CSV")
    trainer.add_argument("--sample-size", type=int, default=1_600_000, help="balanced tweet count; default: 1,600,000 (full dataset). Use optimized hyperparameters for large samples (>500k)")
    trainer.add_argument("--test-size", type=float, default=0.20, help="held-out fraction; default: 0.20")
    trainer.add_argument("--random-state", type=int, default=42, help="reproducibility seed; default: 42")
    trainer.set_defaults(handler=train)
    predictor = commands.add_parser("predict", help="classify a single tweet with the saved model")
    predictor.add_argument("--text", required=True, help="tweet text to classify")
    predictor.set_defaults(handler=predict)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.handler(args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
