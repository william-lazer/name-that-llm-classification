
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

# 2. Prepare reproducible validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

splits = list(
    splitter.split(X, y, groups=groups)
)

scores = []

# 3. Train and evaluate on each split
for split_number, (train_idx, valid_idx) in enumerate(
    splits, start=1
):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Word-level TF-IDF
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=100000,
        sublinear_tf=True,
        strip_accents="unicode"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    # LinearSVC with the best C found so far
    model = LinearSVC(
        C=0.5,
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

    print(
        f"Split {split_number}: "
        f"Balanced Accuracy = {score:.4f} "
        f"({score * 100:.2f}%)"
    )

# 4. Summarize performance
mean_score = sum(scores) / len(scores)

print("\n========== WORD-LEVEL LINEARSVC ==========")
print(f"Mean Balanced Accuracy: {mean_score:.4f}")
print(f"Mean Percentage: {mean_score * 100:.2f}%")
print(f"Highest Score: {max(scores) * 100:.2f}%")
print(f"Lowest Score: {min(scores) * 100:.2f}%")
