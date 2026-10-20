import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score


# 1. Load the dataset
train = pd.read_csv("data/train.csv")

X = train["text"]
y = train["model"]
groups = train["prompt_id"]


# 2. Configure the same three splits for both models
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

results = {
    "Logistic Regression": [],
    "LinearSVC": []
}


# 3. Compare the classifiers
for split_number, (train_idx, valid_idx) in enumerate(
    splitter.split(X, y, groups=groups),
    start=1
):

    X_train = X.iloc[train_idx]
    X_valid = X.iloc[valid_idx]

    y_train = y.iloc[train_idx]
    y_valid = y.iloc[valid_idx]

    # Fit the vectorizer on training text only
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100000,
        sublinear_tf=True
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_valid_tfidf = vectorizer.transform(X_valid)

    # Model 1: Logistic Regression
    lr_model = LogisticRegression(
        max_iter=1500,
        C=5.0
    )

    lr_model.fit(X_train_tfidf, y_train)

    lr_predictions = lr_model.predict(X_valid_tfidf)

    lr_score = balanced_accuracy_score(
        y_valid,
        lr_predictions
    )

    results["Logistic Regression"].append(lr_score)

    # Model 2: LinearSVC
    svm_model = LinearSVC(
        C=1.0,
        class_weight="balanced",
        random_state=42
    )

    svm_model.fit(X_train_tfidf, y_train)

    svm_predictions = svm_model.predict(X_valid_tfidf)

    svm_score = balanced_accuracy_score(
        y_valid,
        svm_predictions
    )

    results["LinearSVC"].append(svm_score)

    print(f"\nSplit {split_number}")
    print(f"Logistic Regression: {lr_score:.4f}")
    print(f"LinearSVC:           {svm_score:.4f}")


# 4. Compare average performance
print("\n========== FINAL COMPARISON ==========")

for model_name, scores in results.items():
    average_score = sum(scores) / len(scores)

    print(
        f"{model_name}: "
        f"Mean Balanced Accuracy = "
        f"{average_score:.4f} ({average_score * 100:.2f}%)"
    )