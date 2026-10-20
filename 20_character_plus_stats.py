
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin


# 1. Custom transformer to extract text statistics
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


# 2. Load dataset
train = pd.read_csv("data/train.csv")

X = train["text"].fillna("").astype(str)
y = train["model"]
groups = train["prompt_id"]

print("Dataset shape:", train.shape)
print("Number of classes:", y.nunique())


# 3. Grouped validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

splits = list(splitter.split(X, y, groups=groups))

scores = []


# 4. Evaluate the combined feature model
for split_number, (train_idx, valid_idx) in enumerate(
    splits, start=1
):

    print(f"\nTraining split {split_number}...")

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Character TF-IDF
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_valid_char = char_vectorizer.transform(X_valid)

    # Numeric text statistics
    stats_transformer = TextStatistics()

    X_train_stats = stats_transformer.fit_transform(X_train)
    X_valid_stats = stats_transformer.transform(X_valid)

    # Scale numeric features using training data only
    scaler = StandardScaler()

    X_train_stats = scaler.fit_transform(X_train_stats)
    X_valid_stats = scaler.transform(X_valid_stats)

    # Give the statistics a modest initial weight
    X_train_stats = csr_matrix(X_train_stats * 0.5)
    X_valid_stats = csr_matrix(X_valid_stats * 0.5)

    # Combine text and numeric features
    from scipy.sparse import hstack

    X_train_combined = hstack(
        [X_train_char, X_train_stats],
        format="csr"
    )

    X_valid_combined = hstack(
        [X_valid_char, X_valid_stats],
        format="csr"
    )

    # Train LinearSVC
    model = LinearSVC(
        C=0.5,
        class_weight="balanced",
        max_iter=10000,
        random_state=42
    )

    model.fit(X_train_combined, y_train)

    predictions = model.predict(X_valid_combined)

    score = balanced_accuracy_score(
        y_valid,
        predictions
    )

    scores.append(score)

    print(
        f"Split {split_number}: "
        f"Balanced Accuracy = {score:.4f} "
        f"({score * 100:.2f}%)"
    )


# 5. Report results
mean_score = sum(scores) / len(scores)

print("\n========== CHARACTER + STATISTICS ==========")
print(f"Mean Balanced Accuracy: {mean_score:.4f}")
print(f"Mean Percentage: {mean_score * 100:.2f}%")
print(f"Highest Score: {max(scores) * 100:.2f}%")
print(f"Lowest Score: {min(scores) * 100:.2f}%")

print("\nPrevious best mean: 29.94%")

print(
    "Difference: "
    f"{(mean_score - 0.2994) * 100:+.2f} percentage points"
)