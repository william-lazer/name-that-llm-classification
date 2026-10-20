
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
train = pd.read_csv("data/train.csv")

X = train["text"].fillna("").astype(str)
y = train["model"]
groups = train["prompt_id"]

# 2. Reproduce the first validation split
splitter = GroupShuffleSplit(
    n_splits=3,
    test_size=0.20,
    random_state=42
)

splits = list(splitter.split(X, y, groups=groups))
train_idx, valid_idx = splits[0]

X_train = X.iloc[train_idx]
X_valid = X.iloc[valid_idx]

y_train = y.iloc[train_idx]
y_valid = y.iloc[valid_idx]

# 3. Transform text using character TF-IDF
vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    max_features=100000,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_valid_tfidf = vectorizer.transform(X_valid)

# 4. Train our current best model
model = LinearSVC(
    C=0.5,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_valid_tfidf)

# 5. Overall evaluation
score = balanced_accuracy_score(y_valid, predictions)

print("\n========== MODEL PERFORMANCE ==========")
print(f"Balanced Accuracy: {score:.4f}")
print(f"Balanced Accuracy (%): {score * 100:.2f}%")

# 6. Per-class precision, recall and F1-score
print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_valid,
        predictions,
        labels=sorted(y.unique()),
        zero_division=0
    )
)

# 7. Confusion matrix
labels = sorted(y.unique())

cm = confusion_matrix(
    y_valid,
    predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=labels,
    columns=labels
)

print("\n========== CONFUSION MATRIX ==========")
print("Rows = Actual model")
print("Columns = Predicted model")
print(cm_df.to_string())

# 8. Save the confusion matrix for further inspection
cm_df.to_csv("confusion_matrix.csv")

print("\nConfusion matrix saved to confusion_matrix.csv")