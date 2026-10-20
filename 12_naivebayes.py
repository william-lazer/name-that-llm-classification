import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import balanced_accuracy_score


# 1. Load the dataset
train = pd.read_csv("data/train.csv")

X = train["text"]
y = train["model"]
groups = train["prompt_id"]


# 2. Create three grouped validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

scores = []


# 3. Train and evaluate Naive Bayes
for split_number, (train_idx, valid_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Convert text into word-level TF-IDF features
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=50000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    # Train Multinomial Naive Bayes
    model = MultinomialNB(alpha=0.1)

    model.fit(X_train_tfidf, y_train)

    # Predict validation labels
    predictions = model.predict(X_valid_tfidf)

    # Evaluate balanced accuracy
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
print("\n========== NAIVE BAYES RESULTS ==========")

print(f"Mean Balanced Accuracy: {sum(scores) / len(scores):.4f}")
print(f"Highest Score: {max(scores):.4f}")
print(f"Lowest Score: {min(scores):.4f}")