import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix, hstack
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin


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


# 1. Load the training data
train = pd.read_csv("data/train.csv")

X = train["text"].fillna("").astype(str)
y = train["model"]
groups = train["prompt_id"]

print("Dataset shape:", train.shape)
print("Number of classes:", y.nunique())


# 2. Create fixed validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

splits = list(splitter.split(X, y, groups=groups))


# 3. Test numerical feature weights
weights = [0.25, 0.50, 0.75, 1.00]

results = []

for weight in weights:

    scores = []

    print(f"\nTesting statistics weight: {weight}")

    for split_number, (train_idx, valid_idx) in enumerate(
        splits, start=1
    ):

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

        # Numerical text statistics
        stats_transformer = TextStatistics()

        X_train_stats = stats_transformer.fit_transform(X_train)
        X_valid_stats = stats_transformer.transform(X_valid)

        # Fit scaler using training data only
        scaler = StandardScaler()

        X_train_stats = scaler.fit_transform(X_train_stats)
        X_valid_stats = scaler.transform(X_valid_stats)

        # Apply the experimental weight
        X_train_stats = csr_matrix(X_train_stats * weight)
        X_valid_stats = csr_matrix(X_valid_stats * weight)

        # Combine features
        X_train_combined = hstack(
            [X_train_char, X_train_stats],
            format="csr"
        )

        X_valid_combined = hstack(
            [X_valid_char, X_valid_stats],
            format="csr"
        )

        # Train the classifier
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
            f"  Split {split_number}: "
            f"{score * 100:.2f}%"
        )

    mean_score = np.mean(scores)

    results.append({
        "statistics_weight": weight,
        "split_1": scores[0],
        "split_2": scores[1],
        "split_3": scores[2],
        "mean_balanced_accuracy": mean_score
    })


# 4. Display and save results
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "mean_balanced_accuracy",
    ascending=False
)

print("\n========== WEIGHT COMPARISON ==========")

print(
    results_df.to_string(
        index=False,
        formatters={
            "split_1": "{:.4f}".format,
            "split_2": "{:.4f}".format,
            "split_3": "{:.4f}".format,
            "mean_balanced_accuracy": "{:.4f}".format
        }
    )
)

results_df.to_csv(
    "statistics_weight_results.csv",
    index=False
)

print("\nResults saved to statistics_weight_results.csv")