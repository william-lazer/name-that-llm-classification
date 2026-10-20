import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score


class TextStatistics(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        if isinstance(X, pd.Series):
            texts = X.fillna("").astype(str)
        else:
            texts = pd.Series(X).fillna("").astype(str)

        features = []

        for text in texts:
            features.append([
                len(text),
                len(text.split()),
                text.count("?"),
                text.count("!"),
                text.count("."),
                text.count(","),
                text.count(":"),
                text.count(";"),
                text.count("'"),
                text.count('"'),
                text.count("("),
                text.count(")"),
                text.count("-"),
                text.count("\n"),
            ])

        return np.asarray(features, dtype=np.float64)


# 1. Load data
train = pd.read_csv("data/train.csv")

X = train["text"].fillna("").astype(str)
y = train["model"]
groups = train["prompt_id"]

print("Dataset shape:", train.shape)
print("Number of classes:", y.nunique())


# 2. Generate identical validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

splits = list(splitter.split(X, y, groups=groups))

results = {
    "Character TF-IDF only": [],
    "Character TF-IDF + statistics": []
}


# 3. Compare feature configurations
for split_number, (train_idx, valid_idx) in enumerate(
    splits, start=1
):

    print(f"\nTraining split {split_number}...")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Fit character TF-IDF on training text only
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    X_train_char = vectorizer.fit_transform(X_train)
    X_valid_char = vectorizer.transform(X_valid)

    # Model A: Character TF-IDF only
    model_a = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=10000,
        random_state=42
    )

    model_a.fit(X_train_char, y_train)

    predictions_a = model_a.predict(X_valid_char)

    score_a = balanced_accuracy_score(
        y_valid, predictions_a
    )

    results["Character TF-IDF only"].append(score_a)

    # Extract numerical text statistics
    stats_transformer = TextStatistics()

    X_train_stats = stats_transformer.fit_transform(X_train)
    X_valid_stats = stats_transformer.transform(X_valid)

    # Fit scaler on training statistics only
    scaler = StandardScaler()

    X_train_stats = scaler.fit_transform(X_train_stats)
    X_valid_stats = scaler.transform(X_valid_stats)

    # Apply the same numerical feature weight as the baseline
    X_train_stats = csr_matrix(X_train_stats * 0.5)
    X_valid_stats = csr_matrix(X_valid_stats * 0.5)

    # Model B: Character TF-IDF + statistics
    X_train_combined = hstack(
        [X_train_char, X_train_stats],
        format="csr"
    )

    X_valid_combined = hstack(
        [X_valid_char, X_valid_stats],
        format="csr"
    )

    model_b = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=10000,
        random_state=42
    )

    model_b.fit(X_train_combined, y_train)

    predictions_b = model_b.predict(X_valid_combined)

    score_b = balanced_accuracy_score(
        y_valid, predictions_b
    )

    results["Character TF-IDF + statistics"].append(score_b)

    print(f"Character TF-IDF only:       {score_a * 100:.2f}%")
    print(f"Character TF-IDF + stats:    {score_b * 100:.2f}%")


# 4. Summarize the comparison
print("\n========== FEATURE COMPARISON ==========")

for model_name, scores in results.items():

    mean_score = np.mean(scores)

    print(
        f"{model_name}: "
        f"{mean_score * 100:.2f}% mean balanced accuracy"
    )

mean_a = np.mean(results["Character TF-IDF only"])
mean_b = np.mean(results["Character TF-IDF + statistics"])

print(
    f"\nDifference from adding statistics: "
    f"{(mean_b - mean_a) * 100:+.2f} percentage points"
)

comparison = pd.DataFrame({
    "split": [1, 2, 3],
    "character_tfidf_only": results["Character TF-IDF only"],
    "character_tfidf_plus_statistics": results[
        "Character TF-IDF + statistics"
    ]
})

comparison.to_csv("feature_comparison_results.csv", index=False)

print("\nResults saved to feature_comparison_results.csv")