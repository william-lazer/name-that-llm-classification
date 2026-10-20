# Import Libraries :
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


# 1. Load the training dataset
train = pd.read_csv("data/train.csv")

X = train["text"]
y = train["model"]
groups = train["prompt_id"]


# 2. Configure three grouped validation splits
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

scores = []


# 3. Train and evaluate the model on each split
for split_number, (train_idx, valid_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1
):
    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Fit TF-IDF using training text only
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    # Train the selected model
    model = LogisticRegression(
        max_iter=1500,
        C=5.0
    )

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


# 4. Summarise the results
print("\n--- Final Validation Summary ---")
print(f"Mean Balanced Accuracy: {sum(scores) / len(scores):.4f}")
print(f"Highest Score:          {max(scores):.4f}")
print(f"Lowest Score:           {min(scores):.4f}")