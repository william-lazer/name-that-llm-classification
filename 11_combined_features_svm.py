import pandas as pd

from scipy.sparse import hstack

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score


# 1. Load the dataset
train = pd.read_csv("data/train.csv")

X = train["text"]
y = train["model"]
groups = train["prompt_id"]


# 2. Use the same three grouped splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

scores = []


# 3. Train and evaluate
for split_number, (train_idx, valid_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Word features
    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    X_train_word = word_vectorizer.fit_transform(X_train)
    X_valid_word = word_vectorizer.transform(X_valid)

    # Character features
    char_vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    X_train_char = char_vectorizer.fit_transform(X_train)
    X_valid_char = char_vectorizer.transform(X_valid)

    # Combine the feature matrices
    X_train_combined = hstack(
        [X_train_word, X_train_char]
    ).tocsr()

    X_valid_combined = hstack(
        [X_valid_word, X_valid_char]
    ).tocsr()

    print(
        f"\nSplit {split_number}: "
        f"Combined features = {X_train_combined.shape[1]}"
    )

    # Train LinearSVC
    model = LinearSVC(
        C=1.0,
        class_weight="balanced",
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


# 4. Summarise results
print("\n========== FINAL RESULTS ==========")

print(
    f"Mean Balanced Accuracy: "
    f"{sum(scores) / len(scores):.4f}"
)

print(f"Highest Score: {max(scores):.4f}")
print(f"Lowest Score: {min(scores):.4f}")