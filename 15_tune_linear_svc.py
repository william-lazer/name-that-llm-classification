
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score

# 1. Load dataset
train = pd.read_csv("data/train.csv")

X = train["text"].fillna("").astype(str)
y = train["model"]
groups = train["prompt_id"]

# 2. Prepare three reproducible validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

# Save split indices so every C value uses the same splits
splits = list(splitter.split(X, y, groups=groups))

# 3. Test different regularization strengths
c_values = [0.1, 0.5, 1, 2, 5]

results = []

for c_value in c_values:

    scores = []

    for split_number, (train_idx, valid_idx) in enumerate(
        splits, start=1
    ):

        X_train = X.iloc[train_idx]
        X_valid = X.iloc[valid_idx]

        y_train = y.iloc[train_idx]
        y_valid = y.iloc[valid_idx]

        # Fit TF-IDF on training text only
        vectorizer = TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2,
            max_features=100000,
            sublinear_tf=True
        )

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_valid_tfidf = vectorizer.transform(X_valid)

        # Train LinearSVC
        model = LinearSVC(
            C=c_value,
            class_weight="balanced",
            random_state=42
        )

        model.fit(X_train_tfidf, y_train)

        predictions = model.predict(X_valid_tfidf)

        score = balanced_accuracy_score(
            y_valid,
            predictions
        )

        scores.append(score)

    mean_score = sum(scores) / len(scores)

    results.append({
        "C": c_value,
        "Split 1": scores[0],
        "Split 2": scores[1],
        "Split 3": scores[2],
        "Mean Balanced Accuracy": mean_score
    })

# 4. Display results
results_df = pd.DataFrame(results)

print("\n========== LINEARSVC HYPERPARAMETER TUNING ==========\n")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Split 1": "{:.4f}".format,
            "Split 2": "{:.4f}".format,
            "Split 3": "{:.4f}".format,
            "Mean Balanced Accuracy": "{:.4f}".format
        }
    )
)

best_row = results_df.loc[
    results_df["Mean Balanced Accuracy"].idxmax()
]

print("\n========== BEST RESULT ==========")
print(f"Best C: {best_row['C']}")
print(
    "Mean Balanced Accuracy: "
    f"{best_row['Mean Balanced Accuracy']:.4f}"
)
print(
    "Mean Percentage: "
    f"{best_row['Mean Balanced Accuracy'] * 100:.2f}%"
)